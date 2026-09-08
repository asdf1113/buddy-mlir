# RUN: %PYTHON %s %mlir_runner_utils_dir
# REQUIRES: mpi

import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
LLVM_LIB_DIR = Path(sys.argv[1]).resolve()


def run(
    command: list[str], cwd: Path = SOURCE_ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


with tempfile.TemporaryDirectory(prefix="buddy-deepseek-tp2-make-") as tmp:
    build_dir = Path(tmp)
    configure = run(
        [
            "cmake",
            "-S",
            str(SOURCE_ROOT),
            "-B",
            str(build_dir),
            "-G",
            "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLLVM_DIR={LLVM_LIB_DIR / 'cmake' / 'llvm'}",
            f"-DMLIR_DIR={LLVM_LIB_DIR / 'cmake' / 'mlir'}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON",
            "-DBUDDY_RUNTIME_ENABLE_MPI=ON",
            "-DBUDDY_BUILD_DEEPSEEK_R1_MODEL=OFF",
            "-DBUDDY_BUILD_DEEPSEEK_R1_TP2_MODEL=ON",
            f"-DBUDDY_DSR1_TP2_SPEC={SOURCE_ROOT / 'models' / 'deepseek_r1' / 'specs' / 'f32.json'}",
            "-DBUDDY_RAX_EMBED_PAYLOAD=OFF",
        ]
    )
    assert configure.returncode == 0, configure.stdout + configure.stderr

    graph_path = (
        build_dir
        / "models"
        / "deepseek_r1_tp2"
        / "CMakeFiles"
        / "deepseek_r1_tp2_rax.dir"
        / "build.make"
    )
    graph = graph_path.read_text()
    prefix = "models/deepseek_r1_tp2/"
    stamp = prefix + ".buddy_import_done"
    model = prefix + "deepseek_r1_tp2_model.so"
    plans = [
        prefix + f"layer_partitioned/runtime/rank{rank}_{phase}.json"
        for rank in range(2)
        for phase in ("forward_prefill", "forward_decode")
    ]
    packs = [prefix + f"rank{rank}_params_float32.data" for rank in range(2)]

    assert f"{stamp}: {prefix}generated/config.json" in graph
    assert graph.count("Importing rank-local TP=2 artifacts") == 1
    assert graph.count("import_model.py --config") == 1
    for artifact in plans + packs:
        assert f"{artifact}: {stamp}" in graph, artifact

    shims = prefix + "generated/RaxShims.cpp"
    for consumer in (shims, model):
        assert f"{consumer}: {stamp}" in graph
        for plan in plans:
            assert f"{consumer}: {plan}" in graph

    for rank in range(2):
        rhal = prefix + f"rank{rank}.rhal.mlir"
        rax = prefix + f"rank{rank}.rax"
        assert f"{rhal}: {stamp}" in graph
        for phase in ("forward_prefill", "forward_decode"):
            plan = prefix + f"layer_partitioned/runtime/rank{rank}_{phase}.json"
            assert f"{rhal}: {plan}" in graph
        assert f"{rax}: {rhal}" in graph
        assert f"{rax}: {packs[rank]}" in graph
        assert f"{rax}: {model}" in graph

    cache = (build_dir / "CMakeCache.txt").read_text().splitlines()
    make_program = next(
        line.split("=", 1)[1]
        for line in cache
        if line.startswith("CMAKE_MAKE_PROGRAM:FILEPATH=")
    )
    dry_run = run(
        [
            make_program,
            "-f",
            str(graph_path.relative_to(build_dir)),
            "-n",
            *(prefix + f"rank{rank}.rhal.mlir" for rank in range(2)),
            *packs,
        ],
        cwd=build_dir,
    )
    dry_run_output = dry_run.stdout + dry_run.stderr
    assert "No rule to make target" not in dry_run_output, dry_run_output
    assert dry_run.returncode == 0, dry_run_output
    assert dry_run_output.count("import_model.py --config") == 1, dry_run_output

print("DeepSeek TP2 Unix Makefiles dependency graph checks passed")
