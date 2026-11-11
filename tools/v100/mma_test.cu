extern "C" __global__ void mma_test(const float* a, const float* b, float* c) {
    // Ampere mma.sync.aligned.m16n8k16
    float acc0, acc1, acc2, acc3;
    unsigned a0 = ((const unsigned*)a)[0];
    unsigned b0 = ((const unsigned*)b)[0];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4}, {%5}, {%0,%1,%2,%3};"
        : "=f"(acc0), "=f"(acc1), "=f"(acc2), "=f"(acc3)
        : "r"(a0), "r"(b0), "f"(acc0), "f"(acc1), "f"(acc2), "f"(acc3));
    c[0] = acc0;
}