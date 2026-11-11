#include <cuda.h>
#include <cstdio>
int main(){
  cuInit(0); CUdevice d; cuDeviceGet(&d,0);
  CUcontext cx; cuCtxCreate(&cx,0,d);
  int b0=0,b96=0;
  cuDeviceGetAttribute(&b0, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, d);
  cuDeviceGetAttribute(&b96, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, d);
  printf("perBlock=%d optin=%d\n", b0, b96);
  CUmodule m; CUfunction f;
  CUresult r=cuModuleLoad(&m,"f70.cubin");
  printf("moduleLoad=%d\n",(int)r);
  r=cuModuleGetFunction(&f,m,"attn_decode_gqa_64");
  printf("getFunc=%d\n",(int)r);
  int stat=0,dyn=0,nreg=0;
  cuFuncGetAttribute(&stat, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
  cuFuncGetAttribute(&dyn,  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
  cuFuncGetAttribute(&nreg, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
  printf("gqa_64: static_shared=%d default_maxdyn=%d regs=%d\n", stat, dyn, nreg);
  for(int v : {32<<10,48<<10,64<<10,80<<10,96<<10,102<<10}){
    CUresult s=cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, v);
    printf("  set %dKB -> %d\n", v>>10,(int)s);
  }
  return 0;
}
