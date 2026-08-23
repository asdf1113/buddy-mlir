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
