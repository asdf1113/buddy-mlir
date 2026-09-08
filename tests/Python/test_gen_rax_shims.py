# RUN: %PYTHON %s

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = SOURCE_ROOT / "tools" / "buddy-codegen" / "gen_rax_shims.py"
COMPILE_PIPELINE = (
    SOURCE_ROOT / "tools" / "buddy-codegen" / "compile_pipeline.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_rax_shims", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plan(rank, parameter_name):
    return {
        "graph": "forward_decode",
        "rank": rank,
        "world_size": 2,
        "resources": {
            parameter_name: {
                "shape": [32],
                "dtype": "float32",
                "role": "parameter",
            },
            "input0": {
                "shape": [1, 2, 4],
                "dtype": "float32",
                "role": "input",
            },
            "output0": {
                "shape": [1, 2, 4],
                "dtype": "float32",
                "role": "output",
            },
        },
        "operations": [
            {
                "kind": "dispatch",
                "wrapper": "forward_decode_layer1_seg0",
                "function": "subgraph0_decode1_seg0",
                "parameter_packs": [parameter_name],
                "inputs": ["input0"],
                "outputs": ["output0"],
                "arguments": [parameter_name, "input0", "output0"],
            },
            {
                "kind": "collective",
                "collective": "all_reduce",
                "input": "output0",
                "output": "output0",
            },
            {
                "kind": "dispatch",
                "wrapper": "forward_decode_region0_seg0",
                "function": "subgraph0_decode0_seg0",
                "parameter_packs": [],
                "inputs": ["output0"],
                "outputs": ["input0"],
                "arguments": ["output0", "input0"],
            },
        ],
    }


generator = load_generator()
compile_pipeline_spec = importlib.util.spec_from_file_location(
    "compile_pipeline", COMPILE_PIPELINE
)
assert (
    compile_pipeline_spec is not None
    and compile_pipeline_spec.loader is not None
)
compile_pipeline = importlib.util.module_from_spec(compile_pipeline_spec)
compile_pipeline_spec.loader.exec_module(compile_pipeline)
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    rank0 = root / "rank0.json"
    rank1 = root / "rank1.json"
    rank0.write_text(json.dumps(plan(0, "rank0_params_float32")))
    rank1.write_text(json.dumps(plan(1, "rank1_params_float32")))

    wrappers = generator.load_wrapper_abis([rank0, rank1])
    generated = generator.generate_cpp(wrappers)
    reversed_generated = generator.generate_cpp(
        generator.load_wrapper_abis([rank1, rank0])
    )
    assert generated == reversed_generated

    assert (
        'extern "C" void rax_forward_decode_layer1_seg0(void **args)'
        in generated
    )
    assert (
        'extern "C" void rax_forward_decode_region0_seg0(void **args)'
        in generated
    )
    layer_body = generated.split(
        'extern "C" void rax_forward_decode_layer1_seg0(void **args) {'
    )[1].split("\n}", 1)[0]
    assert layer_body.index("args[0]") < layer_body.index("args[1]")
    assert layer_body.index("args[1]") < layer_body.index("args[2]")
    assert "MemRef1DF32 arg0 = {data0, data0, 0, {32}, {1}};" in layer_body
    assert (
        "MemRef3DF32 arg1 = "
        "{data1, data1, 0, {1, 2, 4}, {8, 4, 1}};" in layer_body
    )
    assert (
        "_mlir_ciface_forward_decode_layer1_seg0(&arg0, &arg1, &arg2);"
        in layer_body
    )

    output = root / "RaxShims.cpp"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--runtime-plan",
            str(rank0),
            "--runtime-plan",
            str(rank1),
            "-o",
            str(output),
        ],
        check=True,
    )
    assert output.read_text() == generated

    default_forward = compile_pipeline.build_stages(
        "forward", 1, "-mcpu=native"
    )
    tp_forward = compile_pipeline.build_stages(
        "forward", 1, "-mcpu=native", tp_wrapper_out_params=True
    )
    standalone_subgraph = compile_pipeline.build_stages(
        "subgraph", 1, "-mcpu=native"
    )
    out_param_pass = "-buffer-results-to-out-params=hoist-static-allocs"
    public_out_param_pass = (
        "-buffer-results-to-out-params="
        "hoist-static-allocs modify-public-functions"
    )
    assert out_param_pass not in default_forward[0][1]
    assert public_out_param_pass not in default_forward[0][1]
    assert tp_forward[0][1][:2] == [out_param_pass, public_out_param_pass]
    assert out_param_pass not in standalone_subgraph[2][1]
    assert public_out_param_pass not in standalone_subgraph[2][1]

    mlir_root = root / "layer_partitioned"
    for rank in range(2):
        phase_dir = mlir_root / f"rank{rank}" / "forward_decode"
        phase_dir.mkdir(parents=True)
        for symbol in (
            "forward_decode_layer1_seg0",
            "forward_decode_region0_seg0",
            "subgraph0_decode1_seg0",
            "subgraph0_decode0_seg0",
        ):
            (phase_dir / f"{symbol}.mlir").write_text(f"// {symbol}\n")
    entries = compile_pipeline.tp_runtime_compile_entries(
        str(mlir_root), [str(rank0), str(rank1)]
    )
    assert len(entries) == 4
    assert all(entry[-1] for entry in entries if entry[3] == "forward")
    assert all(not entry[-1] for entry in entries if entry[3] != "forward")
    assert {
        Path(entry[1]).stem for entry in entries if entry[3] == "forward"
    } == {
        "forward_decode_layer1_seg0",
        "forward_decode_region0_seg0",
    }

    cxx = shutil.which("c++")
    if cxx is not None and sys.platform != "darwin":
        library = root / "libshims.so"
        subprocess.run(
            [cxx, "-shared", "-fPIC", str(output), "-o", str(library)],
            check=True,
        )
        generator.validate_library(wrappers, library)

    malformed = plan(0, "rank0_params_float32")
    malformed["operations"][0]["arguments"] = [
        "input0",
        "rank0_params_float32",
        "output0",
    ]
    malformed_path = root / "malformed.json"
    malformed_path.write_text(json.dumps(malformed))
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--runtime-plan",
            str(malformed_path),
            "-o",
            str(root / "invalid.cpp"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "arguments do not equal parameter_packs + inputs + outputs"
        in result.stderr
    )
