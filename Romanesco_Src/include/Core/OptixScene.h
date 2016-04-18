#ifndef OPTIXSCENE_H
#define OPTIXSCENE_H

#include <optix.h>
#include <sutil.h>
#include <optixu/optixu.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixpp_namespace.h>

using namespace optix;

#include <cuda.h>
#include <cuda_runtime.h>

#include <sutil/Mouse.h>

#include "commonStructs.h"
#include "PinholeCamera.h"


class OptixScene
{
public:
    OptixScene(unsigned int _width, unsigned int _height);
    virtual ~OptixScene();

    virtual void updateBufferSize(unsigned int _width, unsigned int _height);
    virtual void drawToBuffer();
    virtual void createGeometry(std::string _hit_src = "");

    optix::Buffer createGLOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height);
    optix::Buffer createOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height);

    void setCamera(optix::float3 _eye, float _fov, int _width, int _height);
    void setVar(const std::string& _name, float _v);
    void setVar(const std::string& _name, optix::float3 _v);
    void setVar(const std::string& _name, optix::Matrix4x4 _v);

    void setOutputBuffer(std::string _name);

//    InitialCameraData camera_data;
    optix::GeometryGroup m_geometrygroup;

    /* Primary RTAPI objects */
    RTcontext context;
    RTprogram ray_gen_program;
    RTbuffer  m_buffer;

    optix::Context m_context;
    MyPinholeCamera* m_camera;

    /* Parameters */
    RTvariable result_buffer;
    RTvariable draw_color;

    unsigned int m_texId;
    unsigned int vboId;

    float m_time;

    unsigned int   m_rr_begin_depth;
    unsigned int   m_sqrt_num_samples;
    unsigned int   m_frame;
    unsigned int   m_sampling_strategy;

    bool m_camera_changed;

private:
    std::string m_outputBuffer;

};

#endif // OPTIXSCENE_H
