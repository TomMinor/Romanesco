#ifndef OPTIXSCENE_H
#define OPTIXSCENE_H

#include <optix.h>
#include <sutil.h>
#include <optixu/optixu.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>

using namespace optix;

#include <cuda.h>
#include <cuda_runtime.h>

//#include <sutil/Mouse.h>

#include "commonStructs.h"
#include "PinholeCamera.h"

#include "path_tracer/path_tracer.h"

#include <QObject>

class OptixScene : public QObject
{
    Q_OBJECT

    enum class PathTraceRay : unsigned int
    {
        CAMERA   = 0u,
        SHADOW   = 1u,
        BSDF  = 2u
    };

public:
    OptixScene(unsigned int _width, unsigned int _height, QObject *_parent = 0);
    virtual ~OptixScene();

    virtual void updateBufferSize(unsigned int _width, unsigned int _height);
    virtual void drawToBuffer();

    virtual void initialiseScene();
    virtual void createCameras();
    virtual void createWorld();
    virtual void createBuffers();
    virtual void createLights();
    virtual void createLightGeo();

//    virtual void addLight(  )

    virtual void setGeometryHitProgram(std::string _hit_src);
    virtual void setShadingProgram(std::string _hit_src);

    optix::Buffer createGLOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height);
    optix::Buffer createOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height);

    void setCamera(optix::float3 _eye, float _fov, int _width, int _height);
    void setVar(const std::string& _name, float _v);
    void setVar(const std::string& _name, optix::float3 _v);
    void setVar(const std::string& _name, optix::Matrix4x4 _v);

    void setOutputBuffer(std::string _name);

    ///
    /// \brief getBufferContents
    /// \param _name
    /// \return a copy of the buffer contents or null if the buffer doesn't exist
    ///
    float* getBufferContents(std::string _name, RTsize *_elementSize, RTsize *_width, RTsize *_height);

    float* getBufferContents(std::string _name);

    bool saveBuffersToDisk(std::string _filename);

    std::string outputBuffer();

    void setTime(float _t);

//    InitialCameraData camera_data;
    optix::GeometryGroup m_geometrygroup;

    /* Primary RTAPI objects */
    RTcontext context;
    RTprogram ray_gen_program;
    RTbuffer  m_buffer;

    optix::Context m_context;
    PinholeCamera* m_camera;

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

    unsigned int m_width;
    unsigned int m_height;

protected:
    std::string m_outputBuffer;

    std::vector<std::pair<std::string, RTformat>> m_glOutputBuffers;
    std::vector<std::pair<std::string, RTformat>> m_outputBuffers;

    unsigned int m_progressiveTimeout;
    bool m_frameDone;

    std::vector<ParallelogramLight> m_lights;

signals:
    void frameReady();

};

#endif // OPTIXSCENE_H
