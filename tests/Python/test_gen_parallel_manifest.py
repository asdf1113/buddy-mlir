# RUN: %PYTHON %s rax-pack rax-inspect

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_ROOT = Path(
    os.environ.get("BUDDY_SRC_ROOT", Path(__file__).resolve().parents[2])
)
GEN_MANIFEST = SOURCE_ROOT / "tools" / "buddy-codegen" / "gen_manifest.py"
module_spec = importlib.util.spec_from_file_location(
    "buddy_gen_manifest", GEN_MANIFEST
)
manifest_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(manifest_module)


CONFIG = {
    "model_family": "stage4b",
    "model_id": "stage4b_test",
    "shape": {
        "head_num": 1,
        "max_token_len": 4,
        "hidden_size": 2,
        "vocab_size": 8,
        "kv_layers": 1,
    },
    "tokens": {"vocab_file": "vocab.txt"},
    "weights": [
        {
            "tag": "params",
            "mlir_type": "f32",
            "num_elements": 4,
            "file": "params.data",
        }
    ],
    "compilation": {"so_name": "default_kernels.so"},
    "mlir_types": {"kv": "f32", "logits": "f32"},
}

PACK_RESOURCE = "rank0_params_float32"
PACK = {
    "resource": PACK_RESOURCE,
    "dtype": "float32",
    "numel": 4,
    "parameters": [],
}


PREFILL_PLAN = {
    "graph": "prefill_graph",
    "rank": 0,
    "world_size": 2,
    "resources": {
        PACK_RESOURCE: {
            "shape": [4],
            "dtype": "float32",
            "role": "parameter",
        },
        "input0": {"shape": [1, 4], "dtype": "int64", "role": "input"},
        "workspace0": {
            "shape": [2],
            "dtype": "float32",
            "role": "workspace",
        },
        "gathered": {
            "shape": [4],
            "dtype": "float32",
            "role": "workspace",
        },
        "output0": {
            "shape": [2],
            "dtype": "float32",
            "role": "output",
        },
    },
    "values": {},
    "parameter_packs": [PACK],
    "operations": [
        {
            "kind": "dispatch",
            "instance_index": 0,
            "template_id": 0,
            "segment_index": 0,
            "wrapper": "forward_prefill_layer0_seg0",
            "function": "parallel_template0_seg0",
            "parameter_packs": [PACK_RESOURCE],
            "parameter_bindings": [],
            "inputs": ["input0"],
            "outputs": ["workspace0"],
            # Deliberately differs from packs + inputs + outputs. Stage 4B must
            # consume this ABI list directly rather than reconstructing it.
            "arguments": ["input0", PACK_RESOURCE, "workspace0"],
        },
        {
            "kind": "collective",
            "collective": "all_gatherv",
            "input": "workspace0",
            "output": "gathered",
            "recv_counts": [2, 2],
            "displacements": [0, 2],
        },
        {
            "kind": "collective",
            "collective": "reduce_scatter",
            "input": "gathered",
            "output": "output0",
            "recv_counts": [2, 2],
            "reduction": "sum",
        },
    ],
    "runtime_inputs": ["input0"],
    "runtime_outputs": ["output0"],
}


DECODE_PLAN = {
    "graph": "decode_graph",
    "rank": 0,
    "world_size": 2,
    "resources": {
        PACK_RESOURCE: {
            "shape": [4],
            "dtype": "float32",
            "role": "parameter",
        },
        "input0": {"shape": [1, 1], "dtype": "int64", "role": "input"},
        "workspace0": {
            "shape": [4],
            "dtype": "bfloat16",
            "role": "workspace",
        },
        "output0": {
            "shape": [4],
            "dtype": "bfloat16",
            "role": "output",
        },
    },
    "values": {},
    "parameter_packs": [PACK],
    "operations": [
        {
            "kind": "dispatch",
            "instance_index": 0,
            "template_id": 0,
            "segment_index": 0,
            "wrapper": "forward_decode_layer0_seg0",
            "function": "parallel_template0_seg0",
            "parameter_packs": [PACK_RESOURCE],
            "parameter_bindings": [],
            "inputs": ["input0"],
            "outputs": ["workspace0"],
            "arguments": ["input0", "workspace0", PACK_RESOURCE],
        },
        {
            "kind": "collective",
            "collective": "all_reduce",
            "input": "workspace0",
            "output": "workspace0",
            "reduction": "sum",
        },
        {
            "kind": "dispatch",
            "instance_index": 0,
            "template_id": 0,
            "segment_index": 1,
            "wrapper": "forward_decode_layer0_seg1",
            "function": "parallel_template0_seg1",
            "parameter_packs": [PACK_RESOURCE],
            "parameter_bindings": [],
            "inputs": ["workspace0"],
            "outputs": ["output0"],
            "arguments": ["workspace0", PACK_RESOURCE, "output0"],
        },
    ],
    "runtime_inputs": ["input0"],
    "runtime_outputs": ["output0"],
}


plans = {
    "forward_prefill": PREFILL_PLAN,
    "forward_decode": DECODE_PLAN,
}
scheduled = manifest_module.gen_parallel_manifest(
    CONFIG,
    plans,
    dep_shared_libs=["collective_runtime.so"],
    runner_library="runner.so",
    kernel_library="rank0_kernels.so",
)

# Shared packs remain rank-global, while every ordinary resource is scoped to
# its function even when logical names overlap with different tensor types.
assert scheduled.count(f"rhal.constant @{PACK_RESOURCE} ") == 1
assert f'uri = "file:{PACK_RESOURCE}.data"' in scheduled
for symbol in (
    "forward_prefill__input0",
    "forward_prefill__workspace0",
    "forward_prefill__output0",
    "forward_decode__input0",
    "forward_decode__workspace0",
    "forward_decode__output0",
):
    assert f"rhal.buffer @{symbol} " in scheduled
assert "rhal.buffer @input0 " not in scheduled
assert "tensor<1x4xi64>" in scheduled
assert "tensor<4xbf16>" in scheduled

# Wrapper code objects use deterministic unique IDs, one shared kernel URI,
# and the Stage 1/2 host-shim entry naming convention. Dependency IDs follow
# wrapper IDs and use a collision-resistant symbol prefix.
wrapper_blocks = re.findall(
    r"rhal\.codeobj @(scheduled_codeobj__\w+) \{id = (\d+) : i32, "
    r"kind = \"host_shared_lib\",\n"
    r"\s+backend = \"cpu\",\n"
    r"\s+uri = \"([^\"]+)\",\n"
    r"\s+entry_symbol = \"([^\"]+)\"\}",
    scheduled,
)
assert [int(item[1]) for item in wrapper_blocks] == [1, 2, 3]
assert {item[2] for item in wrapper_blocks} == {"file:rank0_kernels.so"}
assert [item[3] for item in wrapper_blocks] == [
    "rax_forward_prefill_layer0_seg0",
    "rax_forward_decode_layer0_seg0",
    "rax_forward_decode_layer0_seg1",
]
assert "@scheduled_runtime_dep_1 {id = 4 : i32" in scheduled

# Function order, body operation order, and dispatch ABI argument order are
# all inherited directly from the input mapping and Stage 4A operations.
assert scheduled.index("rhal.func @forward_prefill") < scheduled.index(
    "rhal.func @forward_decode"
)
assert (
    "rhal.dispatch @scheduled_codeobj__forward_prefill_layer0_seg0 "
    f"[@forward_prefill__input0, @{PACK_RESOURCE}, "
    "@forward_prefill__workspace0]"
) in scheduled
assert (
    "rhal.dispatch @scheduled_codeobj__forward_decode_layer0_seg0 "
    f"[@forward_decode__input0, @forward_decode__workspace0, "
    f"@{PACK_RESOURCE}]"
) in scheduled
decode_body = scheduled[scheduled.index("rhal.func @forward_decode") :]
assert (
    decode_body.index("rhal.dispatch")
    < decode_body.index('kind = "all_reduce"')
    < decode_body.rindex("rhal.dispatch")
)
assert "rhal.collective [@forward_decode__workspace0]" in decode_body

# Out-of-place collective metadata is copied without recomputing it.
assert (
    "rhal.collective [@forward_prefill__workspace0] {\n"
    '      kind = "all_gatherv",\n'
    "      output_buffers = [@forward_prefill__gathered],\n"
    "      recv_counts = array<i64: 2, 2>,\n"
    "      displacements = array<i64: 0, 2>"
) in scheduled
assert (
    "rhal.collective [@forward_prefill__gathered] {\n"
    '      kind = "reduce_scatter",\n'
    "      output_buffers = [@forward_prefill__output0],\n"
    "      recv_counts = array<i64: 2, 2>,\n"
    '      reduction = "sum"'
) in scheduled

# A focused single-function invocation is supported.
single_function = manifest_module.gen_parallel_manifest(
    CONFIG, {"forward_decode": DECODE_PLAN}
)
assert "rhal.func @forward_decode" in single_function
assert "rhal.func @forward_prefill" not in single_function
assert 'uri = "file:default_kernels.so"' in single_function

# The emitter checks only the cross-plan rank/world-size contract and shared
# pack consistency needed to safely form one module.
wrong_rank = copy.deepcopy(DECODE_PLAN)
wrong_rank["rank"] = 1
try:
    manifest_module.gen_parallel_manifest(
        CONFIG,
        {"forward_prefill": PREFILL_PLAN, "forward_decode": wrong_rank},
    )
except ValueError as error:
    assert "has rank 1; expected 0" in str(error)
else:
    raise AssertionError("expected mismatched rank rejection")

wrong_pack = copy.deepcopy(DECODE_PLAN)
wrong_pack["parameter_packs"][0]["numel"] = 5
try:
    manifest_module.gen_parallel_manifest(
        CONFIG,
        {"forward_prefill": PREFILL_PLAN, "forward_decode": wrong_pack},
    )
except ValueError as error:
    assert "inconsistent dtype/size metadata" in str(error)
else:
    raise AssertionError("expected inconsistent parameter-pack rejection")


rax_pack = sys.argv[1] if len(sys.argv) > 1 else shutil.which("rax-pack")
rax_inspect = sys.argv[2] if len(sys.argv) > 2 else shutil.which("rax-inspect")
assert rax_pack is not None, "rax-pack must be available on PATH"
assert rax_inspect is not None, "rax-inspect must be available on PATH"

with tempfile.TemporaryDirectory() as temporary_directory:
    temporary_path = Path(temporary_directory)
    config_path = temporary_path / "config.json"
    prefill_path = temporary_path / "prefill.json"
    decode_path = temporary_path / "decode.json"
    legacy_path = temporary_path / "legacy.mlir"
    scheduled_path = temporary_path / "scheduled.mlir"
    config_path.write_text(json.dumps(CONFIG))
    prefill_path.write_text(json.dumps(PREFILL_PLAN))
    decode_path.write_text(json.dumps(DECODE_PLAN))

    # No --runtime-plan takes the pre-existing attribute-form CLI path.
    subprocess.run(
        [
            sys.executable,
            str(GEN_MANIFEST),
            "--config",
            str(config_path),
            "-o",
            str(legacy_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    legacy = legacy_path.read_text()
    assert legacy == manifest_module.gen_manifest(CONFIG)
    assert 'dispatch = "model_kernels"' in legacy
    assert " body {" not in legacy
    legacy_pack = subprocess.run(
        [rax_pack, str(legacy_path), "-o", str(temporary_path / "legacy.rax")],
        capture_output=True,
        text=True,
    )
    assert legacy_pack.returncode == 0, legacy_pack.stderr

    # The repeatable CLI preserves plan order and produces the same text as
    # the production API before passing through the real RAX tools.
    subprocess.run(
        [
            sys.executable,
            str(GEN_MANIFEST),
            "--config",
            str(config_path),
            "--runtime-plan",
            f"forward_prefill={prefill_path}",
            "--runtime-plan",
            f"forward_decode={decode_path}",
            "--kernel-library",
            "rank0_kernels.so",
            "--dep-shared-lib",
            "collective_runtime.so",
            "--runner-library",
            "runner.so",
            "-o",
            str(scheduled_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert scheduled_path.read_text() == scheduled
    packed_path = temporary_path / "scheduled.rax"
    scheduled_pack = subprocess.run(
        [rax_pack, str(scheduled_path), "-o", str(packed_path)],
        capture_output=True,
        text=True,
    )
    assert scheduled_pack.returncode == 0, scheduled_pack.stderr + scheduled
    inspect_result = subprocess.run(
        [rax_inspect, str(packed_path)],
        capture_output=True,
        text=True,
    )
    assert inspect_result.returncode == 0, inspect_result.stderr
    inspected = inspect_result.stdout

    prefill_inspect = inspected[
        inspected.index("@forward_prefill") : inspected.index("@forward_decode")
    ]
    assert re.findall(r"\[\d+\] (Dispatch|Collective)", prefill_inspect) == [
        "Dispatch",
        "Collective",
        "Collective",
    ]
    assert "Collective kind=AllGatherV" in prefill_inspect
    assert "recv_counts=[2,2] displacements=[0,2]" in prefill_inspect
    assert "Collective kind=ReduceScatter reduction=Sum" in prefill_inspect
    assert "recv_counts=[2,2]" in prefill_inspect

    decode_inspect = inspected[inspected.index("@forward_decode") :]
    assert re.findall(r"\[\d+\] (Dispatch|Collective)", decode_inspect) == [
        "Dispatch",
        "Collective",
        "Dispatch",
    ]
    assert "Collective kind=AllReduce reduction=Sum" in decode_inspect

    malformed = subprocess.run(
        [
            sys.executable,
            str(GEN_MANIFEST),
            "--config",
            str(config_path),
            "--runtime-plan",
            "missing_separator",
        ],
        capture_output=True,
        text=True,
    )
    assert malformed.returncode == 1
    assert (
        "error: malformed --runtime-plan 'missing_separator'; "
        "expected FUNCTION=PATH"
    ) in malformed.stderr
