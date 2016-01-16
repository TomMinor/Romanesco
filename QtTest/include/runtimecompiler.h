#ifndef RUNTIMECOMPILER_H
#define RUNTIMECOMPILER_H

#include <string>
#include <QDebug>

#include <cuda.h>
#include <nvrtc.h>

///@todo Update these to the modern CUDA style stuff
#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      qDebug() << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      qDebug() << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)


class RuntimeCompiler
{
public:
    RuntimeCompiler(const std::string& _name, const std::string _source);
    ~RuntimeCompiler();

private:
    nvrtcProgram m_prog;

    char* m_result;
};

#endif // RUNTIMECOMPILER_H
