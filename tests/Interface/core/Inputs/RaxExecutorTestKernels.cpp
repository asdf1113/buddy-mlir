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
