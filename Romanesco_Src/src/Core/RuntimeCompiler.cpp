#include <fstream>
#include <ostream>

#include "RuntimeCompiler.h"
#include "macrohelpers.h"

#ifndef NVRTC_AVAILABLE
#include <string>
#include <iostream>
#include <cstdio>
#include <memory>
#include <exception>

#include <stdlib.h>
#include <cstring>

std::string exec(const char* cmd) {
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
        throw std::runtime_error("Error opening pipe");
    char buffer[128];
    std::string result = "";
    while (!feof(pipe.get())) {
        if (fgets(buffer, 128, pipe.get()) != NULL)
            result += buffer;
    }
    return result;
}
#endif

RuntimeCompiler::RuntimeCompiler(const std::string &_name, const std::string _source,
                                 std::vector<std::string> _includePaths,
                                 std::vector<std::string> _includeFiles)
    : m_result(nullptr)
{
    std::vector<const char*> opts;
    opts.push_back("--define-macro=ROMANESCO_RUNTIME_COMPILE");
    opts.push_back("--gpu-architecture=compute_20");
    opts.push_back("-rdc=true");
    opts.push_back("-I./kernel");
    // CUDA_INCLUDE_PATH include folder is set in the .pro file at compile time for now
    opts.push_back("-I" MACROTOSTRING(CUDA_INCLUDE_PATH)); // compiler automatically concatenates the string and macro

#ifdef NVRTC_AVAILABLE
    NVRTC_SAFE_CALL( nvrtcCreateProgram(&m_prog, _source.c_str(), _name.c_str(), 0, NULL, NULL) );

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
#else
    const std::string nvccbin = MACROTOSTRING(CUDA_EXE);
    const std::string nvccflags= "--compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -ptx";

    std::string nvcc_opts;
    for(std::string opt : opts)
    {
        nvcc_opts += opt + " ";
    }

    static const std::string tmpFile = "/tmp/out.cu";
    std::ofstream cudaFile(tmpFile, std::ofstream::out );
    cudaFile << _source;
    cudaFile.close();

    const std::string nvcccall = nvccbin + " -m64 " + nvcc_opts + nvccflags + " " + " -o /dev/stdout " + tmpFile;

    std::string result;
    try
    {
        result = exec( nvcccall.c_str() );
    }
    catch( std::runtime_error e)
    {
        qCritical() << e.what();
    }

    m_result = new char[result.length()];
    strcpy(m_result, result.c_str());
#endif
}

RuntimeCompiler::~RuntimeCompiler()
{
    delete m_result;

#ifdef NVRTC_AVAILABLE
    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&m_prog));
#endif
}

