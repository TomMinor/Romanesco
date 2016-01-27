#include "include/runtimecompiler.h"

RuntimeCompiler::RuntimeCompiler(const std::string &_name, const std::string _source,
                                 std::vector<std::string> _includePaths,
                                 std::vector<std::string> _includeFiles)
    : m_result(nullptr)
{
    NVRTC_SAFE_CALL( nvrtcCreateProgram(&m_prog, _source.c_str(), _name.c_str(), 0, NULL, NULL) );

    std::vector<const char*> opts;
    opts.push_back("--gpu-architecture=compute_35");
    opts.push_back("-rdc=true");
    opts.push_back("-I/home/tom/src/Romanesco/QtTest/kernel");
    opts.push_back("-I/usr/local/cuda-7.0/targets/x86_64-linux/include");

    nvrtcResult compileResult = nvrtcCompileProgram(m_prog, opts.size(), opts.data());

    size_t logSize;
    NVRTC_SAFE_CALL( nvrtcGetProgramLogSize(m_prog, &logSize) );
    if (logSize > 1)
    {
        std::vector<char> log(logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(m_prog, &log[0]));
        qDebug() << &log[0];
    }

    if (compileResult != NVRTC_SUCCESS)
    {
        exit(1);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(m_prog, &ptxSize));

    m_result = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(m_prog, m_result));
}

RuntimeCompiler::~RuntimeCompiler()
{
    delete m_result;

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&m_prog));
}

