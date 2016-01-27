TARGET=optixqt
OBJECTS_DIR=obj
 
# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}

UI_DIR=ui
MOC_DIR=moc

CONFIG += c++11
CONFIG += debug_and_releas
CONFIG -= app_bundle

# Force debug for now
QMAKE_CXXFLAGS += -g

QT+=gui opengl core

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

# Whatever sources you want in your program
SOURCES += src/*.cpp \
           src/gui/*.cpp \
           src/SDFOps/*.cpp
 
# Whatever headers you want in your program
HEADERS += include/*.h \
           include/gui/*.h \
           include/SDFOps/*.h

OTHER_FILES += shaders/*

INCLUDEPATH += ./include
INCLUDEPATH += ./include/gui
INCLUDEPATH += ./include/SDFOps

INCLUDEPATH += /opt/local/include
#Whatever libs you want in your program
DESTDIR=./
#Whatever libs you want in your program
CONFIG += console
CONFIG -= app_bundle
 
LIBS += -lpthread -lIlmImf -lHalf

macx:INCLUDEPATH+=/usr/local/include/
unix:LIBS += -L/usr/local/lib
 
#Optix Stuff, so any optix program that we wish to turn into PTX code
CUDA_SOURCES += kernel/*.cu
 
#This will change for you, just set it to wherever you have installed cuda
# Path to cuda SDK install
macx:CUDA_DIR = /Developer/NVIDIA/CUDA-6.5
linux:CUDA_DIR = /usr/local/cuda-7.0

# Path to cuda toolkit install
macx:CUDA_SDK = /Developer/NVIDIA/CUDA-6.5/samples
linux:CUDA_SDK = /usr/local/cuda-7.0/samples
 
# include paths, change this to wherever you have installed OptiX
macx:INCLUDEPATH += /Developer/OptiX/SDK/sutil
macx:INCLUDEPATH += /Developer/OptiX/SDK
linux:INCLUDEPATH += /home/tom/src/optix/SDK/sutil
linux:INCLUDEPATH += /home/tom/src/optix/SDK
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_DIR/common/inc/
INCLUDEPATH += $$CUDA_DIR/../shared/inc/
macx:INCLUDEPATH += /Developer/OptiX/include
linux:INCLUDEPATH += /home/tom/src/optix/include

# lib dirs
#QMAKE_LIBDIR += $$CUDA_DIR/lib64
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_SDK/common/lib
macx:QMAKE_LIBDIR += /Developer/OptiX/lib64
linux:QMAKE_LIBDIR += /home/tom/src/optix/lib64
linux:QMAKE_LIBDIR += /home/tom/src/optix/build/lib

#Add our cuda and optix libraries
LIBS += -lcudart  -loptix -loptixu -lglut -lsutil -lnvrtc
 
# nvcc flags (ptxas option verbose is always useful)
# add the PTX flags to compile optix files
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -ptx
 
#set our ptx directory so that our ptx files are put somewhere else
PTX_DIR = ptx
 
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
 
# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
optix.input = CUDA_SOURCES
 
#Change our output name to something suitable
optix.output = $$PTX_DIR/${QMAKE_FILE_BASE}.cu.ptx
 
# Tweak arch according to your GPU's compute capability
# Either run your device query in cuda/samples or look in section 6 here #http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#axzz3OzHV3KTV
#for optix you can only have one architechture when using the PTX flags when using the -ptx flag you dont want to have the -c flag for compiling
optix.commands = $$CUDA_DIR/bin/nvcc -m64 -gencode arch=compute_50,code=sm_50 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#use this line for debug code
#optix.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -gencode arch=compute_50,code=sm_50 $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#Declare that we wnat to do this before compiling the C++ code
optix.CONFIG = target_predeps
#now declare that we don't want to link these files with gcc, otherwise it will treat them as object #files
optix.CONFIG += no_link
optix.dependency_type = TYPE_C
# Tell Qt that we want add our optix compiler
QMAKE_EXTRA_UNIX_COMPILERS += optix
