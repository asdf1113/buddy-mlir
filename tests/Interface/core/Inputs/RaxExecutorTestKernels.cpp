#include <cstdio>

extern "C" void kernel_a(void **args) {
  auto *value = static_cast<int *>(args[0]);
  auto *increment = static_cast<int *>(args[1]);
  *value += *increment;
}

extern "C" void kernel_b(void **args) {
  auto *factor = static_cast<int *>(args[0]);
  auto *value = static_cast<int *>(args[1]);
  *value *= *factor;
}

extern "C" void deepseek_session_kernel(void **args) {
  auto *parameters = static_cast<int *>(args[0]);
  auto *input = static_cast<int *>(args[1]);
  auto *kvCache = static_cast<int *>(args[2]);
  auto *output = static_cast<int *>(args[3]);
  *output = *parameters + *input + *kvCache;
}

extern "C" void stage5d_rank0_prefill(void **args) {
  *static_cast<float *>(args[0]) = 1.0f;
}

extern "C" void stage5d_rank1_prefill(void **args) {
  *static_cast<float *>(args[0]) = 2.0f;
}

extern "C" void stage5d_rank0_decode(void **args) {
  if (*static_cast<float *>(args[0]) == 1.0f)
    std::puts("Stage 5D.1 rank0 prefill/decode passed");
}

extern "C" void stage5d_rank1_decode(void **args) {
  if (*static_cast<float *>(args[0]) == 2.0f)
    std::puts("Stage 5D.1 rank1 prefill/decode passed");
}
