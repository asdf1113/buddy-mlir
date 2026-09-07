// REQUIRES: mpi, deepseek-r1-model
// RUN: buddy-deepseek-r1-rax-runner-test --prepare %t
// RUN: not buddy-cli --model %t/rank0.rax --tensor-parallel-size 0 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVALID
// RUN: not buddy-cli --model %t/rank0.rax --prompt synthetic \
// RUN:   --max-tokens 0 2>&1 | FileCheck %s --check-prefix=TP1
// RUN: %mpiexec %mpi_numproc_flag 2 %mpi_preflags buddy-cli %mpi_postflags \
// RUN:   --model %t/rank0.rax --tensor-parallel-size 2 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TP2

// INVALID: --tensor-parallel-size must be positive.

// TP1: DeepSeekR1 Inference (buddy-cli / BuddyRuntime)
// TP1: Manifest: {{.*}}rank0.rax
// TP1-NOT: DeepSeek RAX rank

// TP2-DAG: DeepSeek RAX rank 0/2: {{.*}}rank0.rax
// TP2-DAG: DeepSeek RAX rank 1/2: {{.*}}rank1.rax
// TP2-DAG: Stage 5D.1 rank0 prefill/decode passed
// TP2-DAG: Stage 5D.1 rank1 prefill/decode passed
