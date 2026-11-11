// tmp/nr70.cpp - 用 NVRTC 直接编译一份 kernel 源，打印 program log。用于在没有 GPU 的情况下
// 验证某个 kernel 能否编译到指定架构（本仓库的 kernel 全部经 NVRTC 运行时编译，所以这里能复现
// 引擎首次运行时的编译结果）。
// 用法：nr70 <file.cu> [额外选项...]
//   例：nr70 full70.cu                                   → 默认 --gpu-architecture=sm_70
//       nr70 full70.cu -I/usr/local/cuda-12.6/include    → wmma 需要 mma.h 的 include 路径
//       nr70 full80.cu --gpu-architecture=sm_80 -I...    → 对照 Ampere 路径
#include <nvrtc.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <file.cu> [-I dir] [--gpu-architecture=sm_XX]\n", argv[0]); return 2; }
    std::ifstream f(argv[1]);
    if (!f) { std::printf("cannot read %s\n", argv[1]); return 2; }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::vector<std::string> opts;
    bool have_arch = false, have_std = false;
    for (int i = 2; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind("--gpu-architecture", 0) == 0) have_arch = true;
        if (a.rfind("-std", 0) == 0) have_std = true;
        opts.push_back(a);
    }
    if (!have_arch) opts.push_back("--gpu-architecture=sm_70");
    if (!have_std) opts.push_back("-std=c++17");
    std::vector<const char*> op;
    for (const auto& o : opts) op.push_back(o.c_str());

    nvrtcProgram p;
    nvrtcResult r = nvrtcCreateProgram(&p, src.c_str(), "x.cu", 0, nullptr, nullptr);
    if (r != NVRTC_SUCCESS) { std::printf("nvrtcCreateProgram: %s\n", nvrtcGetErrorString(r)); return 1; }
    r = nvrtcCompileProgram(p, (int)op.size(), op.data());

    size_t lg = 0;
    nvrtcGetProgramLogSize(p, &lg);
    std::string log(lg, ' ');
    if (lg > 1) nvrtcGetProgramLog(p, &log[0]);
    std::printf("%s [%s]: %s\n", argv[1], opts[0].c_str(), r == NVRTC_SUCCESS ? "OK" : "FAIL");
    if (r != NVRTC_SUCCESS) std::printf("%s\n", log.substr(0, 1500).c_str());
    else {
        size_t cb = 0;
        if (nvrtcGetCUBINSize(p, &cb) == NVRTC_SUCCESS && cb) std::printf("cubin bytes: %zu\n", cb);
    }
    nvrtcDestroyProgram(&p);
    return r == NVRTC_SUCCESS ? 0 : 1;
}
