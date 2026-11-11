extern "C" __global__ void test(const void* src) {
    extern __shared__ char sm[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        bool ok = true;
        unsigned d = (unsigned)__cvta_generic_to_shared(sm);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
}