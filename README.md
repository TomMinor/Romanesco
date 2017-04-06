# Romanesco
Fractal Renderer


## Build

### Dependencies
* Qt 5
* [CUDA 7.x](https://developer.nvidia.com/cuda-75-downloads-archive) *(tested with 7.5)*
* [Optix 3.8.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) *(other versions are untested)*
* *[Optional]* OpenEXR
* *[Optional]* Boost

### Get the code

```
git clone --recursive https://github.com/TomMinor/Romanesco
```

*Note: Make sure to add --recursive so all the submodules in dependencies/ are synced too*

### Configuring CMake

As specified in the [Qt docs](http://doc.qt.io/qt-5/cmake-manual.html), you need to set CMAKE_PREFIX_PATH to Qt5's location.

If Optix, CUDA, OpenEXR or Boost cannot be found (if they are installed in a non-standard location), their locations can be appended to CMAKE_PREFIX_PATH.

***Note: On Windows it is currently necessary to force a 64-bit build with CMAKE_GENERATOR_PLATFORM***
#### Windows
```batch
mkdir build
cd build
cmake .. -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_PREFIX_PATH="D:\Qt\5.8\msvc2013_64;D:\ProgramData\NVIDIA Corporation\OptiX SDK 3.8.0"
```

#### Linux
```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/Qt/5.8/gcc_64;/opt/NVIDIA-OptiX-SDK-3.8.0-linux64"
```
