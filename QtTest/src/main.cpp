#include "mainwindow.h"
#include <QApplication>
//#include <SDL.h>
//#include <SDL_haptic.h>

#include <QSurfaceFormat>
#include <shaderwindow.h>

#include <mainwindow.h>

#include "nvrtc.h"
#include "cuda.h"

#include <iostream>
#include <vector>
#include <QDebug>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
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
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)


const char* functionSource = "                                  		\n\
__device__  int function(int x, int y, int z)                                 	\n\
{                                                               			\n\
    printf(\"%d, %d, %d\\n\", x, y, z);                                       	\n\
    return 0;							\n\
}                                                               			\n";

const char* funcSource = " \n\
__device__ float ARSE(float a, float b) \n\
{\n\
   return a + b + 27;\n\
}\n\
\n";

void nvrtc_test()
{
    // Create an instance of nvrtcProgram with the SAXPY code string.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, funcSource, "ARSE", 0, NULL, NULL));

    // Compile the program for compute_35.
    const char *opts[] = {"--gpu-architecture=compute_35", "-rdc=true" };
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, opts);

    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1)
    {
        std::vector<char> log(logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, &log[0]));
        std::cout << &log[0] << std::endl;
    }
    if (compileResult != NVRTC_SUCCESS)
        exit(1);

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    qDebug() << ptx;

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    // Load precompiled relocatable source with call to external function
    // and link it together with NVRTC-compiled function.
//    CUlinkState linker;
//    gpuErrchk(cuLinkCreate(0, NULL, NULL, &linker));
//    gpuErrchk(cuLinkAddFile(linker, CU_JIT_INPUT_PTX, "functor.ptx", 0, NULL, NULL));
//    gpuErrchk(cuLinkAddData(linker, CU_JIT_INPUT_PTX, (void*)ptx, ptxSize, "function.ptx", 0, NULL, NULL));
//    void* cubin;
//    gpuErrchk(cuLinkComplete(linker, &cubin, NULL));
//    gpuErrchk(cuModuleLoadDataEx(&module, cubin, 0, NULL, NULL));
//    gpuErrchk(cuLinkDestroy(linker));
}


int main(int argc, char *argv[])
{
    nvrtc_test();

//  if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) < 0 )
//  {
//    // Or die on error
//    qCritical("Unable to initialize SDL");
//  }

//  int numJoyPads = SDL_NumJoysticks();
//  if(numJoyPads ==0) {
//    qWarning( "No joypads found" );
//  } else {
//    qDebug( "Found %d joypads", numJoyPads );
//  }

  QApplication a(argc, argv);
//  MainWindow w;
//  w.show();

  QSurfaceFormat format;
  //format.setVersion(4, 3);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setDepthBufferSize(24);
  format.setStencilBufferSize(8);
  QSurfaceFormat::setDefaultFormat(format);

  //haderWindow window;
  MainWindow window;
  //window.setFormat(format);
  window.resize(800, 600);
  window.show();

  //window.setAnimating(true);

  return a.exec();
}
