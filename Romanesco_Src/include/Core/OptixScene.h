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

#include <future>
#include <iostream>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QWaitCondition>
#include <QDebug>

class OptixScene;

class RenderThread : public QThread
{
    Q_OBJECT

public:
    RenderThread(OptixScene* parent);

    ~RenderThread();

signals:
    void renderedImage();

protected:
    void run() override;

private:
    QMutex mutex;
    QWaitCondition condition;
    OptixScene* m_scene;

//    QSize resultSize;
    bool restart;
    bool abort;
};

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

    void setCamera(optix::float3 _eye, optix::float3 _lookat, float _fov, int _width, int _height);
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

    int2 getResolution();





    bool saveBuffersToDisk(std::string _filename);

    std::string outputBuffer();

    void setTime(float _t);

    void setRelativeTime(float _t);

    int getProgressiveTimeout()
    {
        return m_progressiveTimeout;
    }

    int getNumPixelSamplesSqrt()
    {
        return m_sqrt_num_samples;
    }

    float getNormalDelta()
    {
        return m_context["DEL"]->getFloat();
    }

    float getSurfaceEpsilon()
    {
        return m_context["delta"]->getFloat();
    }

    float getMaximumIterations()
    {
        return m_context["max_iterations"]->getInt();
    }

//    InitialCameraData camera_data;
    optix::GeometryGroup m_geometrygroup;

    /* Primary RTAPI objects */
    RTcontext context;
    RTprogram ray_gen_program;
    RTbuffer  m_buffer;

    optix::Context m_context;
//    PinholeCamera* m_camera;

    /* Parameters */
    RTvariable result_buffer;
    RTvariable draw_color;

    unsigned int m_texId;
    unsigned int vboId;

    float m_time;

    unsigned int   m_rr_begin_depth;
    unsigned int   m_sqrt_num_samples;
    int   m_frame;
    unsigned int   m_sampling_strategy;

    bool m_camera_changed;

    unsigned int m_width;
    unsigned int m_height;

    int m_tileX, m_tileY;

protected:
    std::string m_outputBuffer;

    std::vector<std::pair<std::string, RTformat>> m_glOutputBuffers;
    std::vector<std::pair<std::string, RTformat>> m_outputBuffers;

    int m_progressiveTimeout;
    bool m_frameDone;

    std::vector<ParallelogramLight> m_lights;

    std::future<void> m_future;

private:
    void asyncDraw()
    {
        std::cout << "Drawing" << std::endl;
    }

    void updateGLBuffer();

    RenderThread m_renderThread;

    PinholeCamera* m_camera;

public slots:
    void setProgressiveTimeout(int _timeout)
    {
        m_progressiveTimeout = _timeout + 1;
        if(m_progressiveTimeout < m_frame)
        {
            m_frame = m_progressiveTimeout;
            m_camera_changed = true;
        }
    }

    void setNormalDelta(double _delta)
    {
        m_context[ "DEL" ]->setFloat( _delta );
        m_camera_changed = true;
    }

    void setSurfaceEpsilon(double _epsilon)
    {
        m_context[ "delta" ]->setFloat( _epsilon );
        m_camera_changed = true;
    }

    void setMaximumIterations(int _iterations)
    {
        m_context[ "max_iterations" ]->setUint( _iterations );
        m_camera_changed = true;
    }

    void setSamplesPerPixelSquared(int _samples)
    {
        m_sqrt_num_samples = _samples;
        m_context["sqrt_num_samples"]->setUint( m_sqrt_num_samples );
        m_camera_changed = true;
    }

    void setHorizontalTiles(int _t)
    {
        m_tileX = _t;
    }

    void setVerticalTiles(int _t)
    {
        m_tileY = _t;
    }


signals:
    void frameReady();
    void frameRefined(int _refineFrame);

    void bucketRowReady(uint _row);
    void bucketReady(uint _i, uint _j);

};

#endif // OPTIXSCENE_H
