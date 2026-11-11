#include <cstdio>
#include <cuda.h>
int main(){ CUresult r=cuInit(0); printf("cuInit=%d\n",(int)r); unsigned n=0;
  if(r==CUDA_SUCCESS){ r=cuDeviceGetCount(&n); printf("count=%d rc=%d\n",n,(int)r); }
  return 0;
}
