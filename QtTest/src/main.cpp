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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(CUresult code, const char *file, int line, bool abort=true)
{
   if (code != CUDA_SUCCESS)
   {
       const char* sPtr = new char[512];

       cuGetErrorName(code, &sPtr);
       fprintf(stderr,"GPUassert: %s %s %d\n", sPtr, file, line);
       if (abort) exit(code);
   }
}

#define NVRTC_SAFE_CALL(Name, x)                                             \
  do {                                                                       \
    nvrtcResult result = x;                                                  \
    if (result != NVRTC_SUCCESS) {                                           \
      std::cerr << "\nerror: " << Name << " failed with error " <<           \
                                               nvrtcGetErrorString(result);  \
      exit(1);                                                               \
    }                                                                        \
  } while(0)

void nvrtc_test()
{
    const char* functionSource = "                                  \n\
                             #include <optix.h>                     \n\
    __device__ void function(int x)                                 \n\
    {                                                               \n\
        printf(\"%d\\n\", x);                                       \n\
    }                                                               \n";


    // Create an instance of nvrtcProgram with the SAXPY code string.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL("nvrtcCreateProgram", nvrtcCreateProgram(&prog, functionSource, "function", 0, NULL, NULL));


    //NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -ptx
    //optix.commands = $$CUDA_DIR/bin/nvcc -m64 -gencode arch=compute_50,code=sm_50 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
    ///home/tom/src/optix/include

    // Compile the program for compute_35.
    const char *opts[] = {"--gpu-architecture=compute_35", "-rdc=true",
                          "-I /home/tom/src/optix/include",
                          "-I /usr/local/cuda-7.0/targets/x86_64-linux/include",
                          "-I /home/tom/src/optix/include/internal",


                         };
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 5, opts);

    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL("nvrtcGetProgramLogSize", nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1)
    {
        std::vector<char> log(logSize);
        NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, &log[0]));
        std::cout << &log[0] << std::endl;
    }
    if (compileResult != NVRTC_SUCCESS)
        exit(1);

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL("nvrtcGetPTXSize", nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL("nvrtcGetPTX", nvrtcGetPTX(prog, ptx));

    qDebug() << ptx;

    // Destroy the program.
    NVRTC_SAFE_CALL("nvrtcDestroyProgram", nvrtcDestroyProgram(&prog));

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
//    nvrtc_test();

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
