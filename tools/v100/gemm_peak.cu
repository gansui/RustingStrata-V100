#include <mma.h>
#include <cstdio>
#include <cuda_runtime.h>
using namespace nvcuda;
#define NT 256
#define BM 128
#define BN 128
#define BK 64
#define K0 4096
#define T 100
__global__ void __launch_bounds__(NT) g(const half* a, const half* b, float* c){
  __shared__ __align__(16) half as[BM*BK], bs[BN*BK];
  int bx=blockIdx.x, by=blockIdx.y, tid=threadIdx.x, warp=tid>>5;
  int wm=16*(warp&1), wn=16*((warp>>1)&1);
  wmma::fragment<wmma::accumulator,16,16,16,float> fc;
  wmma::fill_fragment(fc,0.f);
  for(int kb=0;kb<K0;kb+=BK){
    for(int i=tid;i<BM*BK/2;i+=NT){ int r=i/(BK/2),cc=i%(BK/2); ((uint2*)as)[i]=((const uint2*)(a+(size_t)(by*BM+r)*K0+kb))[cc]; }
    for(int i=tid;i<BN*BK/2;i+=NT){ int r=i/(BN/2),cc=i%(BN/2); ((uint2*)bs)[i]=((const uint2*)b)[((size_t)r*K0+bx*BN)/2+cc]; }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> fb;
    wmma::load_matrix_sync(fa,(const half*)(as+wm*BK),BK);
    wmma::load_matrix_sync(fb,(const half*)(bs+wn*BK),BK);
    wmma::mma_sync(fc,fa,fb,fc);
    __syncthreads();
  }
  __shared__ float red[NT];
  wmma::store_matrix_sync(red+warp*256, fc, 16, wmma::mem_row_major);
  __syncthreads();
  if(tid<16){ float s=0; for(int i=0;i<256;i++) s+=red[i]; c[(size_t)by*(4096/16)+bx*16+tid]=s; }
}
int main(){
  half *a,*b; float*c;
  size_t ab=(size_t)4096*4096*2, cb=4096*4;
  cudaMalloc(&a,ab); cudaMalloc(&b,ab); cudaMalloc(&c,cb);
  cudaMemset(a,0x3232,ab); cudaMemset(b,0x4848,ab);
  dim3 gr(4096/BN,4096/BM);
  g<<<gr,NT>>>(a,b,c);
  cudaError_t e=cudaDeviceSynchronize(); if(e){ printf("err %d %s\n",(int)e,cudaGetErrorString(e)); return 1; }
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for(int i=0;i<T;i++) g<<<gr,NT>>>(a,b,c);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms,e0,e1);
  double fl=(double)T*4096.0*4096.0*4096.0*2.0;
  printf("V100 f16 wmma GEMM 4096^3: %.2f ms/iter, %.1f TFLOPS\n", ms/T, fl/ms/1e9);
  return 0;
}
