#include <mma.h>
#include <cstdio>
#include <cuda_runtime.h>
using namespace nvcuda;
#define T 2000
#define BM 128
#define BK 64
#define NT 256
__global__ void __launch_bounds__(NT) gm(const half* x, const half* w, float* y){
  __shared__ __align__(16) half xs[BM*80], ws[64*80];
  int tid=threadIdx.x;
  for(int i=tid;i<BM*BK/2;i+=NT){ int r=i/(BK/2); ((uint2*)xs)[i]=((const uint2*)x)[(size_t)r*(BK/2)+i%(BK/2)]; }
  for(int i=tid;i<64*BK/2;i+=NT){ ((uint2*)ws)[i]=((const uint2*)w)[i]; }
  __syncthreads();
  int warp=tid>>5, lane=tid&31;
  wmma::fragment<wmma::accumulator,16,16,16,float> fc;
  wmma::fill_fragment(fc,0.f);
  int wm=16*(warp&1), wn=16*(warp>>1);
#pragma unroll
  for(int kk=0;kk<BK;kk+=16){
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fb;
    wmma::load_matrix_sync(fa,(const half*)(xs+wm*80+kk),80);
    wmma::load_matrix_sync(fb,(const half*)(ws+wn*80+kk),80);
    wmma::mma_sync(fc,fa,fb,fc);
  }
  wmma::store_matrix_sync(y+warp*256, fc, 16, wmma::mem_row_major);
}
int main(){
  half *x,*w; float*y;
  cudaMalloc(&x,BM*BK*2); cudaMalloc(&w,64*BK*2); cudaMalloc(&y,256*4*256);
  cudaMemset(x,0x1234,BM*BK*2); cudaMemset(w,0x5678,64*BK*2);
  {cudaError_t e=cudaDeviceSynchronize(); if(e) printf("sync1 err %d %s\n",(int)e,cudaGetErrorString(e));}
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  for(int i=0;i<T;i++) gm<<<1,NT>>>(x,w,y);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  double flops=(double)T*BM*64.0*BK*2.0;
  printf("wmma 1 block/sm: %.0f ms, %.0f GFLOPS\n", ms, flops/ms/1e6);
  // 多 SM 版本：grid = 80 SM
  cudaEventRecord(a);
  for(int i=0;i<T;i++) gm<<<80,NT>>>(x,w,y);
  cudaEventRecord(b); cudaEventSynchronize(b);
  cudaEventElapsedTime(&ms,a,b);
  printf("wmma 80 SM: %.3f ms, %.0f GFLOPS\n", ms, flops*80.0/ms/1e6);
  return 0;
}
