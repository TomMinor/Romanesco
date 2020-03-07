#ifndef OPTIXSCENE_H
#define OPTIXSCENE_H

#include "../core/BaseScene.h"

#include "OptixHeaders.h"

//#include <Mouse.h>

#include "commonStructs.h"
#include "PinholeCamera.h"

#include "path_tracer.h"

#include <QObject>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLDebugMessage>
#include <QOpenGLDebugLogger>
#ifdef ROMANESCO_RENDER_WITH_THREADS
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
#endif

class OptixScene : public BaseScene
{
	Q_OBJECT



public:

	OptixScene(unsigned int _width, unsigned int _height, QOpenGLFunctions_4_3_Core* _gl, QObject *_parent = 0);
    virtual ~OptixScene();

    virtual void updateBufferSize(unsigned int _width, unsigned int _height) override;
    virtual void drawToBuffer() override;

    virtual void initialiseScene() override;
    virtual void createCameras() override;
    virtual void createWorld() override;
    virtual void createBuffers() override;
    virtual void createLights() override;
    virtual void createLightGeo() override;

//    virtual void setMaterial(std::string _name);

//    virtual void addLight(  )

    virtual void setGeometryHitProgram(std::string _hit_src) override;
    virtual void setShadingProgram(std::string _hit_src) override;

    void setCurrentMaterial(std::string _name);

    void setCameraType(CameraTypes _type)
    {
        m_cameraMode = _type;
    }

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

	float* getBufferContents(std::string _name, unsigned int *_elementSize);

	optix::int2 getResolution();

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

    CameraTypes m_cameraMode;
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

	optix::GeometryInstance m_geoInstance;

    std::vector<ParallelogramLight> m_lights;

#ifdef ROMANESCO_RENDER_WITH_THREADS
    std::future<void> m_future;
#endif

private:
	QOpenGLFunctions_4_3_Core* m_gl;

    void updateGLBuffer();

#ifdef ROMANESCO_RENDER_WITH_THREADS
    RenderThread m_renderThread;
#endif

    Romanesco::PinholeCamera* m_camera;
	QOpenGLDebugLogger *m_debugLogger;

protected slots:
	void messageLogged(const QOpenGLDebugMessage &msg);

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

    void bucketRowReady(unsigned int _row);
	void bucketReady(unsigned int _i, unsigned int _j);

};

#endif // OPTIXSCENE_H