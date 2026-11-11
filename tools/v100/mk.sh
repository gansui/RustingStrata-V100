#!/bin/bash
cd "$(dirname "$0")" || exit 1   # 产物与相对路径都固定在 tools/v100，不污染仓库根
gen() { # $1=NVRTC_12值 $2=输出文件
cat > "$2" <<HDR
__device__ const unsigned long long iq2xs_grid[512] = {0};
__device__ const unsigned long long iq2xxs_grid[256] = {0};
__device__ const unsigned long long iq2s_grid[1024] = {0};
__device__ const unsigned long long iq1s_grid[2048] = {0};
__device__ const unsigned int iq3xxs_grid[256] = {0};
__device__ const unsigned int iq3s_grid[512] = {0};
__device__ const unsigned char ksigns_iq2xs[128] = {0};
__device__ const signed char kvalues_iq4nl[16] = {0};
#define LLAMA_Q8 0
#define KV_Q8 0
#define NVRTC_12 $1
HDR
cat ../../src/cuda/kernels.cu >> "$2"
}
gen 1 full70.cu
gen 0 full80.cu
