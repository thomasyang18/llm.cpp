#include "reference/flash_attention_1_gpu/forward.hpp"

namespace {
constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
}

