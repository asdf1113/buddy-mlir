# RUN: %PYTHON %s

import os
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(
    os.environ.get("BUDDY_SRC_ROOT", Path(__file__).resolve().parents[2])
)
BUILD_MODEL = SOURCE_ROOT / "tools" / "buddy-codegen" / "build_model.py"
SPEC = SOURCE_ROOT / "models" / "deepseek_r1" / "specs" / "f32.json"


def dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILD_MODEL),
            "--spec",
            str(SPEC),
            "--build-dir",
            "/tmp/buddy-build-model-selection-test",
            "--dry-run",
            *args,
        ],
        cwd=SOURCE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


implicit_tp1 = dry_run()
explicit_tp1 = dry_run("--tensor-parallel-size", "1")
for result in (implicit_tp1, explicit_tp1):
    assert result.returncode == 0, result.stderr
    assert "-DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON" in result.stderr
    assert "-DBUDDY_BUILD_DEEPSEEK_R1_TP2_MODEL=ON" not in result.stderr
    assert "-DBUDDY_RUNTIME_ENABLE_MPI=ON" not in result.stderr
    assert "--target deepseek_r1_rax" in result.stderr

assert implicit_tp1.stderr == explicit_tp1.stderr

tp2 = dry_run("--tensor-parallel-size", "2")
assert tp2.returncode == 0, tp2.stderr
assert "-DBUDDY_BUILD_DEEPSEEK_R1_TP2_MODEL=ON" in tp2.stderr
assert "-DBUDDY_RUNTIME_ENABLE_MPI=ON" in tp2.stderr
assert "-DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON" not in tp2.stderr
assert "--target deepseek_r1_tp2_rax" in tp2.stderr

unsupported = dry_run("--tensor-parallel-size", "4")
assert unsupported.returncode != 0
assert "currently supports only 1 or 2; got 4" in unsupported.stderr

tp1_cmake = (
    SOURCE_ROOT / "models" / "deepseek_r1" / "CMakeLists.txt"
).read_text()
tp2_cmake = (
    SOURCE_ROOT / "models" / "deepseek_r1_tp2" / "CMakeLists.txt"
).read_text()
models_cmake = (SOURCE_ROOT / "models" / "CMakeLists.txt").read_text()
tp1_runner = (
    SOURCE_ROOT / "models" / "deepseek_r1" / "DeepSeekR1Runner.cpp"
).read_text()
model_helper = (
    SOURCE_ROOT / "tools" / "buddy-codegen" / "cmake" / "buddy_model.cmake"
).read_text()

assert "NAME deepseek_r1" in tp1_cmake
assert "add_custom_target(deepseek_r1_tp2_rax" in tp2_cmake
assert "deepseek_r1_tp2_model.so" in tp2_cmake
assert "deepseek_r1_tp2_runner.so" in tp2_cmake
assert 'set(_TP2_BIN "${CMAKE_CURRENT_BINARY_DIR}")' in tp2_cmake
assert "add_subdirectory(deepseek_r1)" in models_cmake
assert "add_subdirectory(deepseek_r1_tp2)" in models_cmake
assert "models/deepseek_r1/specs/f32.json" in tp2_cmake
assert not (SOURCE_ROOT / "models" / "deepseek_r1_tp2" / "specs").exists()
assert "DeepSeekR1Rax" not in tp1_cmake
assert "DeepSeekR1Rax" not in tp1_runner
assert "BUDDY_RUNTIME_ENABLE_MPI" not in tp1_cmake
assert "BUDDY_RUNTIME_ENABLE_MPI" not in tp1_runner
assert "MDL_PARALLEL_RUNTIME" not in model_helper
assert "BUDDY_DEEPSEEK_R1_TENSOR_PARALLEL_SIZE" not in model_helper
assert ".tmp/stage" not in tp2_cmake

print("build_model DeepSeek TP selection checks passed")
