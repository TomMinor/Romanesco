#define GL_GLEXT_PROTOTYPES

#ifdef _WIN32
#include <windows.h>
#endif

///@Todo put in a nice header
#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif


#include "ImageWriter.h"
#include "OptixScene.h"
#include "RuntimeCompiler.h"
#include "RenderMath.h"

#define USE_DEBUG_EXCEPTIONS 0

#include <algorithm>
#include "Base_SDFOP.h"
#include "Primitive/Sphere_SDFOP.h"
#include "DomainOp/Transform_SDFOP.h"
#include <glm/gtc/matrix_transform.hpp>


#include <QWindow>
//#include <QOpenGLFunctions>
#include <QOpenGLPaintDevice>
#include <QOpenGLFramebufferObject>
#include <QScreen>
#include <QDebug>
#include <QKeyEvent>
//#include <QtMath>
#include <QDir>
#include <QCoreApplication>

#ifndef _WIN32
#include <unistd.h>
#endif

//#include <math.h>
#include <fstream>
#include <string>
#include <cerrno>
#include <assert.h>
#include <iostream>
#include <stdexcept>

#ifdef BOOST__AVAILABLE
#include <boost/algorithm/string/join.hpp>
#endif

//#include <ImageLoader.h>

#include "stringutilities.h"
#include "sutil.h"
#include <ImageLoader.h>

///@todo
/// * Split this into a simple base class and derive from that, OptixScene -> OptixSceneAdaptive -> OptixScenePathTracer
/// * All camera stuff should be moved into it's own, simpler class

#ifdef ROMANESCO_RENDER_WITH_THREADS
RenderThread::RenderThread(OptixScene* parent)
    : QThread( static_cast<QObject*>(parent) ), m_scene(parent)
{
    restart = false;
    abort = false;
}

RenderThread::~RenderThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();

    wait();
}

void RenderThread::run()
{
    forever
    {
        mutex.lock();
        // Reinit
        if(abort)
        {
            break;
        }

//        m_scene->drawToBuffer();

        mutex.unlock();

        emit renderedImage();

        mutex.lock();

        qDebug() << "Render Thread";

//        if (!restart)
//        {
//            condition.wait(&mutex);
//        }

        restart = false;
        mutex.unlock();
    }
}
#endif

static int timeoutFunc()
{
    QCoreApplication::processEvents();

    qDebug() << "TIMEOUT TEST";
    return 0;
}


OptixScene::OptixScene(unsigned int _width, unsigned int _height, QOpenGLFunctions_4_3_Core* _gl, QObject *_parent)
	: QObject(_parent), m_time(0.0f), /* m_renderThread(this),*/ m_camera(nullptr), m_gl(_gl)
{
#ifndef NDEBUG
	m_debugLogger = new QOpenGLDebugLogger(this);
	if (m_debugLogger->initialize())
	{
		m_debugLogger->setObjectName("OptixScene");
		qDebug() << "OptixScene OpenGL Debug Logger" << m_debugLogger << "\n";
		connect(m_debugLogger, SIGNAL(messageLogged(QOpenGLDebugMessage)), this, SLOT(messageLogged(QOpenGLDebugMessage)));
		m_debugLogger->startLogging();
	}
#endif // NDEBUG

    /// ================ Initialise Output Texture Buffer ======================
    m_gl->glGenTextures( 1, &m_texId );
	m_gl->glBindTexture(GL_TEXTURE_2D, m_texId);

    // Change these to GL_LINEAR for super- or sub-sampling
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	m_gl->glBindTexture(GL_TEXTURE_2D, 0);
    /// --------------------------------------------------------------

    /// ================ Initialise Context ======================
    m_context = optix::Context::create();

    m_context->setRayTypeCount( 3 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize( 1800 );
    m_context->setEntryPointCount( static_cast<unsigned int>(CameraTypes::TOTALCAMERATYPES) );

#if USE_DEBUG_EXCEPTIONS
    // Disable this by default for performance, otherwise the stitched PTX code will have lots of exception handling inside.
    m_context->setPrintEnabled(true);
    m_context->setPrintLaunchIndex(256, 256); // Launch index (0,0) at lower left.
    m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif
    /// --------------------------------------------------------------

    /// ================ Initialise Output Buffers ======================
    createBuffers();
    updateBufferSize(_width, _height);
    setOutputBuffer("output_buffer");
    /// --------------------------------------------------------------

    /// ===================== Initialise World ======================
    initialiseScene();
    createCameras();
    // Create scene geo
    createLights();
    createWorld();
//    createLightGeo();

    /// --------------------------------------------------------------

	try
	{
		m_context->validate();
		m_context->compile();
	}
	catch (optix::Exception &e)
	{
		qWarning() << e.what();
		throw e;
	}
    

    m_progressiveTimeout = 10;

//    m_future = std::async( std::launch::async, &OptixScene::asyncDraw, this );

//    m_renderThread.start(QThread::LowPriority);

    m_frame = 1;

    m_tileX = m_tileY = 2;

    m_cameraMode = CameraTypes::PINHOLE;

    m_context->setTimeoutCallback( timeoutFunc, 1  );
}


void OptixScene::messageLogged(const QOpenGLDebugMessage &msg)
{
	QString error = "(OptixScene) ";

	// Format based on severity
	switch (msg.severity())
	{
	case QOpenGLDebugMessage::NotificationSeverity:
		error += "--";
		break;
	case QOpenGLDebugMessage::HighSeverity:
		error += "!!";
		break;
	case QOpenGLDebugMessage::MediumSeverity:
		error += "!~";
		break;
	case QOpenGLDebugMessage::LowSeverity:
		error += "~~";
		break;
	}

	error += " (";

	// Format based on source
#define CASE(c) case QOpenGLDebugMessage::c: error += #c; break
	switch (msg.source())
	{
		CASE(APISource);
		CASE(WindowSystemSource);
		CASE(ShaderCompilerSource);
		CASE(ThirdPartySource);
		CASE(ApplicationSource);
		CASE(OtherSource);
		CASE(InvalidSource);
	}
#undef CASE

	error += " : ";

	// Format based on type
#define CASE(c) case QOpenGLDebugMessage::c: error += #c; break
	switch (msg.type())
	{
		CASE(ErrorType);
		CASE(DeprecatedBehaviorType);
		CASE(UndefinedBehaviorType);
		CASE(PortabilityType);
		CASE(PerformanceType);
		CASE(OtherType);
		CASE(MarkerType);
		CASE(GroupPushType);
		CASE(GroupPopType);
	}
#undef CASE

	error += ")";
	qDebug() << qPrintable(error) << "\n" << qPrintable(msg.message()) << "\n";
}

void OptixScene::createBuffers()
{
    m_glOutputBuffers.clear();

    m_glOutputBuffers.push_back( std::make_pair("output_buffer",				RT_FORMAT_FLOAT4) );

    m_glOutputBuffers.push_back( std::make_pair("output_buffer_nrm",			RT_FORMAT_FLOAT3) );
    m_glOutputBuffers.push_back( std::make_pair("output_buffer_world",			RT_FORMAT_FLOAT3) );
    m_glOutputBuffers.push_back( std::make_pair("output_buffer_diffuse",		RT_FORMAT_FLOAT3) );
    m_glOutputBuffers.push_back( std::make_pair("output_buffer_trap",			RT_FORMAT_FLOAT3) );

    m_glOutputBuffers.push_back( std::make_pair("output_buffer_iteration",		RT_FORMAT_FLOAT) );
    m_glOutputBuffers.push_back( std::make_pair("output_buffer_depth",			RT_FORMAT_FLOAT) );

    // Everything is gl for the sake of visualisation right now, but maybe we'd want to add export only buffers later
    m_outputBuffers.clear();
}

optix::Buffer OptixScene::createGLOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height)
{
    optix::Buffer buffer;

    GLuint vbo = 0;
	m_gl->glGenBuffers(1, &vbo);
	m_gl->glBindBuffer(GL_ARRAY_BUFFER, vbo);

    size_t element_size;
    m_context->checkError( rtuGetSizeForRTformat(_format, &element_size) );
	m_gl->glBufferData(GL_ARRAY_BUFFER, element_size * _width * _height, 0, GL_STREAM_DRAW);
	m_gl->glBindBuffer(GL_ARRAY_BUFFER, 0);

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
    buffer->setFormat(_format);
    buffer->setSize( _width, _height );

    return buffer;
}

optix::Buffer OptixScene::createOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height)
{
    optix::Buffer buffer;
    buffer = m_context->createBuffer(RT_BUFFER_OUTPUT);
    buffer->setFormat(_format);
    buffer->setSize( _width, _height );

    return buffer;
}

optix::int2 OptixScene::getResolution()
{
    optix::Buffer buffer = m_context[m_outputBuffer]->getBuffer();
    RTsize width, height;
    buffer->getSize(width, height);
    return optix::make_int2(width, height);
}

void OptixScene::setTime(float _t)
{
    m_time = _t;
    m_context[ "global_t" ]->setFloat( m_time );
}

void OptixScene::setRelativeTime(float _t)
{
    m_context[ "relative_t" ]->setFloat( _t );
}

void OptixScene::setCamera(optix::float3 _eye, optix::float3 _lookat, float _fov, int _width, int _height)
{
    m_camera->setParameters( _eye,
                             _eye + optix::make_float3(0,0,1),
							 optix::make_float3(0.0f, 1.0f, 0.0f),
                             _fov, // hfov is ignored when using keep vertical
                             _fov,
                             Romanesco::PinholeCamera::KeepVertical );
    m_camera->setAspectRatio( static_cast<float>(_width)/static_cast<float>(_height) );

    optix::float3 eye, U, V, W;
//    float aspectRatio = static_cast<float>(m_width)/static_cast<float>(m_height);
//    float inputAngle = atan( radians(_fov * 0.5) );
//    float outputAngle = degrees(2.0f * atanf(aspectRatio * tanf(radians(0.5f * (inputAngle)))) );

    ///@todo Make this more physically accurate
    /// http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/how-pinhole-camera-works-part-2
//    float focalLength = aspectRatio;

    eye = _eye;
    m_camera->getEyeUVW(eye, U, V, W);

    m_context["eye"]->setFloat( eye );
    m_context["U"]->setFloat( optix::make_float3(1, 0, 0) );
    m_context["V"]->setFloat( optix::make_float3(0, 1, 0) );
    m_context["W"]->setFloat( optix::make_float3(0, 0, 1) );

    m_context["U"]->setFloat( U );
    m_context["V"]->setFloat( V );
    m_context["W"]->setFloat( W );

	optix::float3 ulen = optix::make_float3(0, 0, 1) * tanf(radians(_fov*0.5f));
	optix::float3 camera_u = optix::make_float3(1, 0, 0) * ulen;
	optix::float3 vlen = optix::make_float3(0, 0, 1) * tanf(radians(_fov*0.5f));
	optix::float3 camera_v = optix::make_float3(0, 1, 0) * vlen;

//    m_context["U"]->setFloat( camera_u );
//    m_context["V"]->setFloat( camera_v );
//    m_context["W"]->setFloat( W );

//    float focalLength = _width / (2.0f * tan(_fov / 2.0f));
//    focalLength = degrees(2.0f * atanf(static_cast<float>(_width)/static_cast<float>(_height) * tanf(radians(0.5f * _fov))));
//    m_context["W"]->setFloat( optix::make_float3(0, 0, 1) * focalLength );

    m_camera_changed = true;
}

void OptixScene::setOutputBuffer(std::string _name)
{
    m_outputBuffer = _name;
}

void OptixScene::setVar(const std::string& _name, float _v)
{
    m_context[_name]->setFloat(_v);
}

void OptixScene::setVar(const std::string& _name, optix::float3 _v )
{
    m_context[_name]->setFloat( _v  );
}

void OptixScene::setVar(const std::string& _name, optix::Matrix4x4 _v )
{
    m_context[_name]->setMatrix4x4fv(false, _v.getData());
}


void OptixScene::updateBufferSize(unsigned int _width, unsigned int _height)
{
    m_width =_width;
    m_height = _height;

    // Update any GL bound Optix buffers
    for( auto& buffer : m_glOutputBuffers )
    {
        std::string bufferName = buffer.first;
        if(m_context[bufferName]->getType() == RT_OBJECTTYPE_UNKNOWN )
        {
            RTformat bufferType = buffer.second;
            m_context[bufferName]->set( createGLOutputBuffer(bufferType, _width, _height) );
        }
        else
        {
            m_context[bufferName]->getBuffer()->setSize(_width, _height);

            m_context[bufferName]->getBuffer()->unregisterGLBuffer();
			m_gl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_context[bufferName]->getBuffer()->getGLBOId());
			m_gl->glBufferData(GL_PIXEL_UNPACK_BUFFER, m_context[bufferName]->getBuffer()->getElementSize() * _width * _height, 0, GL_STREAM_DRAW);
			m_gl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            m_context[bufferName]->getBuffer()->registerGLBuffer();
        }
    }

    // Update regular Optix buffers
    for( auto& buffer : m_outputBuffers )
    {
        std::string bufferName = buffer.first;
        if(m_context[bufferName]->getType() == RT_OBJECTTYPE_UNKNOWN )
        {
            RTformat bufferType = buffer.second;
            m_context[bufferName]->set( createOutputBuffer(bufferType, _width, _height) );
        }
        else
        {
            m_context[bufferName]->getBuffer()->setSize(_width, _height);
        }
    }

    m_frame = 1;
    m_frameDone = false;
}

///
/// \brief createAreaLight Taken from the Optix path tracer demo
/// \param m_context
/// \param m_pgram_bounding_box
/// \param m_pgram_intersection
/// \param anchor
/// \param offset1
/// \param offset2
/// \return
///
optix::GeometryInstance createAreaLight(optix::Context* m_context,
                                      optix::Program* m_pgram_bounding_box,
                                      optix::Program* m_pgram_intersection,
									  const optix::float3& anchor,
									  const optix::float3& offset1,
									  const optix::float3& offset2)
{
	optix::Geometry parallelogram = (*m_context)->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( *m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( *m_pgram_bounding_box );

  optix::float3 normal = optix::normalize(optix::cross(offset1, offset2));
  float d = optix::dot(normal, anchor);
  optix::float4 plane = optix::make_float4(normal, d);

  optix::float3 v1 = offset1 / optix::dot(offset1, offset1);
  optix::float3 v2 = offset2 / optix::dot(offset2, offset2);

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );

  optix::GeometryInstance gi = (*m_context)->createGeometryInstance();
  gi->setGeometry(parallelogram);
  return gi;
}

void setMaterial(optix::GeometryInstance& gi,
	optix::Material material,
                                   const std::string& color_name,
								   const optix::float3& color)
{
  gi->addMaterial(material);
  gi[color_name]->setFloat(color);
}


void OptixScene::initialiseScene()
{
    ///@todo How many of these are even used

//    m_context["scene_epsilon"]->setFloat( 1.e-4f );
    m_context["scene_epsilon"]->setFloat( 1.e-3f );
    m_context["max_depth"]->setInt( 5 );
    m_context["color_t"]->setFloat( 0.0f );
    m_context["shadowsActive"]->setUint( 0u );

    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );
    // Miss program
    //m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( "ptx/constantbg.cu.ptx", "miss" ) );
//    m_context["bg_color"]->setFloat( optix::make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );
    m_context["bg_color"]->setFloat( optix::make_float3(0.1f) );

    setTime(0.0f);
}

void OptixScene::createLights()
{
    // Setup lights
    m_context["ambient_light_color"]->setFloat(0.1f,0.1f,0.3f);

	optix::float3 test_data[] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f }
    };

    optix::Buffer test_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, sizeof(test_data)/sizeof(test_data[0]) );
    memcpy( test_buffer->map(), test_data, sizeof(test_data) );
    test_buffer->unmap();

    m_context["test"]->set(test_buffer);


    {
        glm::mat4 rot = glm::mat4(1.0f);
        glm::rotate(rot, 30.0f, glm::vec3(1,0,0));

        glm::vec3 v1(-130.0f, 0.0f, 0.0f);
//        v1 = glm::vec3(glm::vec4(v1, 1.0) * rot);
        glm::vec3 v2( 0.0f, 0.0f, 105.0f);
//        v2 = glm::vec3(glm::vec4(v2, 1.0) * rot);

        ParallelogramLight light;
//        light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
		light.corner = optix::make_float3(0.0f, 300.0f, 0.0f);
        //    light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
        //    light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
		light.v1 = optix::make_float3(v1.x, v1.y, v1.z);
		light.v2 = optix::make_float3(v2.x, v2.y, v2.z);
		light.normal = optix::normalize(optix::cross(light.v1, light.v2));
		light.emission = optix::make_float3(3.0f);

        m_lights.push_back(light);
    }

    {
        glm::mat4 rot = glm::mat4(1.0f);
        glm::rotate(rot, 180.0f, glm::vec3(1,0,0));

        glm::vec3 v1(-130.0f, 0.0f, 0.0f);
//        v1 = glm::vec3(glm::vec4(v1, 1.0) * rot);
        glm::vec3 v2( 0.0f, 0.0f, 105.0f);
//        v2 = glm::vec3(glm::vec4(v2, 1.0) * rot);

        ParallelogramLight light;
//        light.corner   = make_float3( 343.0f, -148.6f, 227.0f);
		light.corner = optix::make_float3(0.0f, -300.0f, 0.0f);
        //    light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
        //    light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
		light.v1 = optix::make_float3(v1.x, v1.y, v1.z);
		light.v2 = optix::make_float3(v2.x, v2.y, v2.z);
		light.normal = optix::normalize(optix::cross(light.v1, light.v2));
		light.emission = optix::make_float3(20.0f, 10.0f, 2.5f);

        m_lights.push_back(light);
    }

    {
        glm::mat4 rot = glm::mat4(1.0f);
        glm::rotate(rot, 30.0f, glm::vec3(1,0,0));

        glm::vec3 v1(0.0f, 0.0f, -130.0f);
//      v1 = glm::vec3(glm::vec4(v1, 1.0) * rot);
        glm::vec3 v2( 0.0f, 105.0f, 0.0f);
//      v2 = glm::vec3(glm::vec4(v2, 1.0) * rot);

        ParallelogramLight light;
//      light.corner   = optix::make_float3( 343.0f, 548.6f, 227.0f );
		light.corner = optix::make_float3(100.0f, -50.0f, 0.0f);
//      light.v1       = optix::make_float3( -130.0f, 0.0f, 0.0f );
//      light.v2       = optix::make_float3( 0.0f, 0.0f, 105.0f );
		light.v1 = optix::make_float3(v1.x, v1.y, v1.z);
		light.v2 = optix::make_float3(v2.x, v2.y, v2.z);
		light.normal = optix::normalize(optix::cross(light.v1, light.v2));
		light.emission = optix::make_float3(.5f) * 13.0f;

        m_lights.push_back(light);
    }

    {
        glm::mat4 rot = glm::mat4(1.0f);
        glm::rotate(rot, 30.0f, glm::vec3(1,0,0));

        glm::vec3 v1(0.0f, 0.0f, -130.0f);
//      v1 = glm::vec3(glm::vec4(v1, 1.0) * rot);
        glm::vec3 v2( 0.0f, 105.0f, 0.0f);
//      v2 = glm::vec3(glm::vec4(v2, 1.0) * rot);

        ParallelogramLight light;
//      light.corner   = optix::make_float3( 343.0f, 548.6f, 227.0f );
		light.corner = optix::make_float3(100.0f, -50.0f, 130.0f);
//      light.v1       = optix::make_float3( -130.0f, 0.0f, 0.0f );
//      light.v2       = optix::make_float3( 0.0f, 0.0f, 105.0f );
		light.v1 = optix::make_float3(v1.x, v1.y, v1.z);
		light.v2 = optix::make_float3(v2.x, v2.y, v2.z);
		light.normal = optix::normalize(optix::cross(light.v1, light.v2));
		light.emission = optix::make_float3(.5f) * 13.0f;

        m_lights.push_back(light);
    }

	optix::Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( m_lights.size() );
    memcpy( light_buffer->map(), &m_lights[0], sizeof(ParallelogramLight) * m_lights.size() );
    light_buffer->unmap();
    m_context["lights"]->setBuffer( light_buffer );
}

void OptixScene::createCameras()
{
    m_rr_begin_depth = 1u;
    m_sqrt_num_samples = 1u;
    m_camera_changed = true;

    m_context["pathtrace_ray_type"]->setUint( static_cast<unsigned int>(PathTraceRay::CAMERA) );
    m_context["pathtrace_shadow_ray_type"]->setUint( static_cast<unsigned int>(PathTraceRay::SHADOW) );
    m_context["pathtrace_bsdf_shadow_ray_type"]->setUint( static_cast<unsigned int>(PathTraceRay::BSDF) );
    m_context["rr_begin_depth"]->setUint(m_rr_begin_depth);

//    camera_data = InitialCameraData( optix::make_float3( 3.0f, 2.0f, -3.0f ), // eye
//                                 optix::make_float3( 0.0f, 0.3f,  0.0f ), // lookat
//                                 optix::make_float3( 0.0f, 1.0f,  0.0f ), // up
//                                 60.0f );                          // vfov

	m_camera = new Romanesco::PinholeCamera(
		optix::make_float3(0.0f, 0.0f, 0.0f),
				   optix::make_float3(0.0f, 0.0f, 1.0f),
				   optix::make_float3(0.0f, 1.0f, 0.0f),
                   -1.0f, // hfov is ignored when using keep vertical
                   60.0f,
				   Romanesco::PinholeCamera::KeepVertical);

    optix::Buffer buffer = m_context[m_outputBuffer]->getBuffer();
    RTsize width, height;
    buffer->getSize(width, height);
	setCamera(optix::make_float3(0.0f), optix::make_float3(0.0f, 0.3f, 0.0f), 60.0f, width, height);

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );

    // Setup path tracer
    m_context["sqrt_num_samples"]->setUint( m_sqrt_num_samples );
    m_context["frame_number"]->setUint(1);

    // Index of sampling_stategy (BSDF, light, MIS)
    m_sampling_strategy = 0;
    m_context["sampling_stategy"]->setInt(m_sampling_strategy);

	/// @todo Fix relative paths
    optix::Program ray_gen_program = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "pathtrace_camera" );
    optix::Program exception_program = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "exception" );
    m_context->setRayGenerationProgram(  static_cast<unsigned int>( CameraTypes::PINHOLE ) , ray_gen_program );
    m_context->setExceptionProgram(  static_cast<unsigned int>( CameraTypes::PINHOLE ), exception_program );

    optix::Program environment_ray_gen_program = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "env_camera" );
    m_context->setRayGenerationProgram(  static_cast<unsigned int>( CameraTypes::ENVIRONMENT ), environment_ray_gen_program );
    m_context->setExceptionProgram(  static_cast<unsigned int>( CameraTypes::ENVIRONMENT ), exception_program );

    // Miss programs
    m_context->setMissProgram( static_cast<unsigned int>(PathTraceRay::CAMERA), m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "envmap_miss" ) );

    const optix::float3 default_color = m_context["bg_color"]->getFloat3();
	
	/// @todo Fix absolute path
	m_context["envmap"]->setTextureSampler(loadTexture(m_context, qPrintable(QDir::currentPath() + QString("/hdr/CedarCity.hdr")), default_color));
}

void OptixScene::createLightGeo()
{
	optix::GeometryGroup geo = m_context["top_shadower"]->getGeometryGroup();

	optix::GeometryGroup lights = m_context->createGeometryGroup();

	optix::Program m_pgram_bounding_box = m_context->createProgramFromPTXFile("ptx/parallelogram.cu.ptx", "bounds");
	optix::Program m_pgram_intersection = m_context->createProgramFromPTXFile("ptx/parallelogram.cu.ptx", "intersect");

	optix::Material emissiveMat = m_context->createMaterial();
	optix::Program diffuse_emitter = m_context->createProgramFromPTXFile("ptx/menger.cu.ptx", "diffuseEmitter");
    emissiveMat->setClosestHitProgram( static_cast<unsigned int>(PathTraceRay::CAMERA), diffuse_emitter );

	const optix::float3 light_em = optix::make_float3(15.0f, 15.0f, 5.0f);

	std::vector<optix::GeometryInstance> areaLights;

    // Light
    areaLights.push_back( createAreaLight( &m_context,
                                        &m_pgram_bounding_box,
                                        &m_pgram_intersection,
										optix::make_float3(-2500, 2000.0, -2500),
										optix::make_float3(5000.0f, 0.0f, 0.0f),
										optix::make_float3(0.0f, 0.0f, 5000.0f)));
    setMaterial(areaLights.back(), emissiveMat, "emission_color", light_em);

//    lights->setChildCount( areaLights.size() );
    for(unsigned int i = 0; i < areaLights.size(); i++)
    {
//        lights->setChild(i, areaLights[i]);
        geo->addChild( areaLights[i] );
    }
}

void OptixScene::setCurrentMaterial(std::string _name)
{
	const optix::float3 white = optix::make_float3(0.8f, 0.8f, 0.8f);

	optix::Material diffuseMat = m_context->createMaterial();
	optix::Program diffuse_closestHit = m_context->createProgramFromPTXFile("ptx/menger.cu.ptx", _name);
    diffuseMat->setClosestHitProgram( static_cast<unsigned int>(PathTraceRay::CAMERA), diffuse_closestHit );

	optix::Program diffuse_anyHit = m_context->createProgramFromPTXFile("ptx/menger.cu.ptx", "shadow");
    diffuseMat->setAnyHitProgram( static_cast<unsigned int>(PathTraceRay::SHADOW), diffuse_anyHit );

    setMaterial(m_geoInstance, diffuseMat, "diffuse_color", white);

    m_context->validate();
    m_context->compile();
}

void OptixScene::createWorld()
{
    ///@todo Optix error checking
    optix::Geometry SDF_scene = m_context->createGeometry();
    SDF_scene->setPrimitiveCount( 1u );
    SDF_scene->setBoundingBoxProgram( m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "bounds" ) );
    SDF_scene->setIntersectionProgram( m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "intersect" ) );

    m_geoInstance = m_context->createGeometryInstance();
    m_geoInstance->setGeometry(SDF_scene);


	/// @todo Sort out this material stuff, why was some of it disabled
    optix::Material diffuseMat = m_context->createMaterial();
	optix::Program diffuse_closestHit = m_context->createProgramFromPTXFile("ptx/menger.cu.ptx", "pathtrace_diffuse");
    diffuseMat->setClosestHitProgram( static_cast<unsigned int>(PathTraceRay::CAMERA), diffuse_closestHit );

	optix::Program diffuse_anyHit = m_context->createProgramFromPTXFile("ptx/menger.cu.ptx", "shadow");
    diffuseMat->setAnyHitProgram( static_cast<unsigned int>(PathTraceRay::SHADOW), diffuse_anyHit );

    setMaterial(m_geoInstance, diffuseMat, "diffuse_color", optix::make_float3(1.0,1.0,1.0) );

	optix::GeometryGroup m_geometrygroup = m_context->createGeometryGroup();
    m_geometrygroup->addChild(m_geoInstance);



	optix::GeometryGroup lights = m_context->createGeometryGroup();

	optix::Program m_pgram_bounding_box = m_context->createProgramFromPTXFile("ptx/parallelogram.cu.ptx", "bounds");
	optix::Program m_pgram_intersection = m_context->createProgramFromPTXFile("ptx/parallelogram.cu.ptx", "intersect");

	optix::Material emissiveMat = m_context->createMaterial();
	optix::Program diffuse_emitter = m_context->createProgramFromPTXFile("ptx/menger.cu.ptx", "diffuseEmitter");
    emissiveMat->setClosestHitProgram( static_cast<unsigned int>(PathTraceRay::CAMERA), diffuse_emitter );

	const optix::float3 light_em = optix::make_float3(15.0f, 15.0f, 5.0f);

	std::vector<optix::GeometryInstance> areaLights;

    // Light
//    areaLights.push_back( createAreaLight( &m_context,
//                                        &m_pgram_bounding_box,
//                                        &m_pgram_intersection,
//                                        make_float3( -2500, 2000.0, -2500),
//                                        make_float3( 5000.0f, 0.0f, 0.0f),
//                                        make_float3( 0.0f, 0.0f, 5000.0f) ) );
    for(ParallelogramLight light : m_lights)
    {
        areaLights.push_back( createAreaLight( &m_context,
                                            &m_pgram_bounding_box,
                                            &m_pgram_intersection,
                                            light.corner,
                                            light.v1,
                                            light.v2 ) );
        setMaterial(areaLights.back(), emissiveMat, "emission_color", light_em);
    }

//    lights->setChildCount( areaLights.size() );
//    for(unsigned int i = 0; i < areaLights.size(); i++)
//    {
////        lights->setChild(i, areaLights[i]);
//        m_geometrygroup->addChild( areaLights[i] );
//    }

    m_geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
    m_context["top_object"]->set( m_geometrygroup );

    // Create shadow group (no light)
	optix::GeometryGroup shadow_group = m_context->createGeometryGroup();
    shadow_group->addChild(m_geoInstance);
    shadow_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
    m_context["top_shadower"]->set( shadow_group );

    float  m_alpha;
    float  m_delta;
    float m_DEL;
    unsigned int m_max_iterations;

//    m_alpha = 0.003f;
//    m_delta = 0.001f;
//    m_DEL = 0.001f;
//    m_max_iterations = 20;

    m_delta = 0.01f;
    m_DEL = 0.001f;
    m_max_iterations = 20;

    m_context[ "delta" ]->setFloat( m_delta );
    m_context[ "max_iterations" ]->setUint( m_max_iterations );
    m_context[ "DEL" ]->setFloat( m_DEL );

    std::string defaultSrc = R"(\
        #include "romanescocore.h"

        HIT_PROGRAM float4 hit(float3 x, int maxIterations, float global_t)
        {
            Mandelbulb sdf(maxIterations);
            sdf.evalParameters();
            sdf.setTime(global_t);

            float p = (3.0 * abs(sin(global_t / 40.0))) + 5.0;
            sdf.setPower( p );

            return make_float4( sdf.evalDistance(x), sdf.getTrap0(), sdf.getTrap1(), sdf.getTrap2() );
        }

            )";

            setGeometryHitProgram(defaultSrc);

        //setCurrentMaterial("pathtrace_diffuse");
}


//#define DEMO

void OptixScene::setGeometryHitProgram(std::string _hit_src)
{
    static std::string geometryhook_src = R"(
#include "cutil_math.h"

__device__ __noinline__ float3 distancehit_hook()
{
     return make_float3(1,0,0);
}

)";

    std::string hit_src = _hit_src;//(_hit_src == "") ? geometryhook_src : _hit_src;

    // Compile function source to ptx
    RuntimeCompiler program("hit", _hit_src);
    try
    {
        program.compile();
    }
    catch(std::runtime_error& e)
    {
        qWarning() << e.what();
        return;
    }

    const char* ptx = program.getResult();
    if(!ptx)
    {
        qWarning() << "Ptx source is empty";
        return;
    }

    optix::Program testcallable;
    try
    {
        testcallable = m_context->createProgramFromPTXString(ptx, "hit");
    } catch(optix::Exception &e)
    {
        qWarning("Failed to create program from ptx: %s", e.getErrorString().c_str() );
        return;
    }

    try
    {
        m_context["hit_hook"]->set(testcallable);
    } catch(optix::Exception &e)
    {
        qWarning("Failed to create program from ptx: %s", e.getErrorString().c_str() );
        return;
    }


}

void OptixScene::setShadingProgram(std::string _hit_src)
{
    static std::string shadinghook_src = R"(
#include "cutil_math.h"

__device__ __noinline__ float3 shade_hook()
{
     return make_float3(1,1,1);
}

)";

    std::string hit_src = (_hit_src == "") ? shadinghook_src : _hit_src;

    // Compile function source to ptx
    RuntimeCompiler program("shade_hook", _hit_src);
    try
    {
        program.compile();
    }
    catch(std::runtime_error& e)
    {
        qWarning() << e.what();
        return;
    }

    const char* ptx = program.getResult();
    if(!ptx)
    {
        qWarning() << "Ptx source is empty";
        return;
    }

    try
    {
        optix::Program testcallable = m_context->createProgramFromPTXString(ptx, "distancehit_hook");
        m_context["do_work"]->set(testcallable);
	}
	catch (optix::Exception &e)
    {
        qWarning("Failed to create program from ptx");
        return;
    }
}

OptixScene::~OptixScene()
{

}

float* OptixScene::getBufferContents(std::string _name, RTsize* _elementSize, RTsize* _width, RTsize* _height)
{
    // Calculate the buffer byte size etc from it's optix buffer properties
    optix::Buffer buffer = m_context[_name]->getBuffer();
    RTsize buffer_width, buffer_height;
    buffer->getSize(buffer_width, buffer_height);

    (*_elementSize) = buffer->getElementSize();
    (*_width) = buffer_width;
    (*_height) = buffer_height;

    RTsize bufferSize = buffer_width * buffer_height;
    float* hostPtr = new float[buffer->getElementSize() * bufferSize];
    CUdeviceptr devicePtr = buffer->getDevicePointer( 0 );
	cudaMemcpy((void*)hostPtr, (void*)devicePtr, buffer->getElementSize() * bufferSize, cudaMemcpyDeviceToHost);
//    qDebug() << buffer->getElementSize() * bufferSize;

    return hostPtr;
}

float* OptixScene::getBufferContents(std::string _name, unsigned int *_elementSize)
{
    RTsize ignore, width, height;

	float* tmp = getBufferContents(_name, &ignore, &width, &height);
	*_elementSize = ignore;

	return tmp;
}

std::string OptixScene::outputBuffer()
{
    return m_outputBuffer;
}

bool OptixScene::saveBuffersToDisk(std::string _filename)
{
    #ifdef OPENEXR_AVAILABLE
	std::map<std::string, std::string> buffers = {
			{ "",	"output_buffer" },
			{ "N",	"output_buffer_nrm" },
			{ "P",	"output_buffer_world" },
			{ "d",	"output_buffer_diffuse" },
			//{ "Z",	"output_buffer_depth" },
			//{ "i",	"output_buffer_iteration" },
			{ "t",	"output_buffer_trap" }
	};

	RTsize buffer_width, buffer_height;
	m_context["output_buffer"]->getBuffer()->getSize(buffer_width, buffer_height);
	const unsigned int totalPixels = static_cast<unsigned int>(buffer_width)* static_cast<unsigned int>(buffer_height);

	std::vector<Romanesco::Channel> channels;
	unsigned int width = static_cast<unsigned int>(buffer_width);
	unsigned int height = static_cast<unsigned int>(buffer_height);

	for (auto& buffer : buffers)
	{
		qDebug() << buffer.first.c_str();

		unsigned int elementSize = 0;
		float* pixels = getBufferContents(buffer.second, &elementSize);

		channels.push_back(Romanesco::Channel(pixels, elementSize, width, height, buffer.first));

		delete [] pixels;
		pixels = nullptr;
	}

    //float* rgba = getBufferContents("output_buffer");
    //float* normal = getBufferContents("output_buffer_nrm");
    //float* world = getBufferContents("output_buffer_world");
    //float* diffuse = getBufferContents("output_buffer_diffuse");
    //float* depth = getBufferContents("output_buffer_depth");
    //float* iteration = getBufferContents("output_buffer_iteration");
    //float* trap = getBufferContents("output_buffer_trap");


	
	//channels.reserve(totalPixels);

	
	
	//channels.push_back(Romanesco::Channel(normal, width, height, "nrm"));
	//channels.push_back(Romanesco::Channel(world, width, height, "P"));
	//channels.push_back(Romanesco::Channel(diffuse, width, height, "diffuse"));
	//channels.push_back(Romanesco::Channel(depth, width, height, "z"));
	//channels.push_back(Romanesco::Channel(iteration, width, height, "iter"));
	//channels.push_back(Romanesco::Channel(trap, width, height, "trap"));

    ///@Todo move this into one loop, messing up my pointer arithmetic when i do so right now
    /*for(unsigned int i = 0; i < totalPixels * 4; i+=4)
    {
        ImageWriter::Pixel tmp;
        tmp.r = rgba[i + 0];
        tmp.g = rgba[i + 1];
        tmp.b = rgba[i + 2];
        tmp.a = rgba[i + 3];

        pixels.push_back(tmp);
    }*/

	/// @todo EXR channels
    //for(unsigned int j = 0; j < totalPixels; j++)
    //{
    //    pixels[j].x_nrm = normal[3*j + 0];
    //    pixels[j].y_nrm = normal[3*j + 1];
    //    pixels[j].z_nrm = normal[3*j + 2];

    //    pixels[j].x_pos = world[3*j + 0];
    //    pixels[j].y_pos = world[3*j + 1];
    //    pixels[j].z_pos = world[3*j + 2];

    //    pixels[j].trapR = trap[3*j + 0];
    //    pixels[j].trapG = trap[3*j + 1];
    //    pixels[j].trapB = trap[3*j + 2];

    //    pixels[j].diffuseR = diffuse[3*j +0];
    //    pixels[j].diffuseG = diffuse[3*j +1];
    //    pixels[j].diffuseB = diffuse[3*j +2];
    //}

    //for(unsigned int k = 0; k < totalPixels; k++)
    //{
    //    pixels[k].z = depth[k];

    //    pixels[k].iteration = iteration[k];
    //}


    //delete [] diffuse;
    //delete [] normal;
    //delete [] world;
    //delete [] depth;
    //delete [] trap;

    //diffuse = nullptr;
    //normal = nullptr;
    //world = nullptr;
    //depth = nullptr;
    //trap = nullptr;

    ImageWriter image(_filename, buffer_width, buffer_height);
    return image.write(channels);
    #endif
    return false;
}

void OptixScene::updateGLBuffer()
{
    // Copy optix buffer to gl texture directly on the GPU
    // (current visible buffer could potentially be changed even when optix scene is finished rendering, so copy over this every frame regardless)
    /// ==================  Copy to texture =======================
    optix::Buffer buffer = m_context[m_outputBuffer]->getBuffer();
    RTformat buffer_format = buffer->getFormat();
    RTsize buffer_width, buffer_height;
    buffer->getSize( buffer_width, buffer_height );

    vboId = buffer->getGLBOId();

    if (vboId)
    {
		m_gl->glBindTexture(GL_TEXTURE_2D, m_texId);

        // send pbo to texture
		m_gl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vboId);

        RTsize elementSize = buffer->getElementSize();
        if      ((elementSize % 8) == 0) m_gl->glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
		else if ((elementSize % 4) == 0) m_gl->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		else if ((elementSize % 2) == 0) m_gl->glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
		else                             m_gl->glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        {
            if(buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
				m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
            } else if(buffer_format == RT_FORMAT_FLOAT4) {
				m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
            } else if(buffer_format == RT_FORMAT_FLOAT3) {
				m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
            } else if(buffer_format == RT_FORMAT_FLOAT) {
				m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
            } else {
                assert(0 && "Unknown buffer format");
            }
        }

		m_gl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        assert(0 && "Couldn't bind GL Buffer Object");
    }
}



void OptixScene::drawToBuffer()
{
    if( m_camera_changed ) {
        m_camera_changed = false;
        m_frame = 1;
        m_frameDone = false;
    }

    RTsize buffer_width, buffer_height;
    m_context["output_buffer"]->getBuffer()->getSize( buffer_width, buffer_height );

    // http://heart-touching-graphics.blogspot.co.uk/2012/04/bidirectional-path-tracing-using-nvidia_27.html
    // https://devtalk.nvidia.com/default/topic/806609/optix/splitting-work-on-multiple-launches/
    // http://graphics.cs.aueb.gr/graphics/docs/Constantinos%20Kalampokis%20Thesis.pdf
//    int2 NoOfTiles = make_int2(12, 12);
	optix::int2 NoOfTiles = optix::make_int2(m_tileX, m_tileY);
	optix::float2 launch_index_tileSize = optix::make_float2(float(buffer_width) / NoOfTiles.x,
                                                float(buffer_height) / NoOfTiles.y );

    bool isFrameReady = false;
    // Update Optix scene if necessary
    if( m_frame < m_progressiveTimeout )
    {
        m_context["frame_number"]->setUint(m_frame);
        m_frame++;
        m_context["TileSize"]->setFloat( launch_index_tileSize );

        for(int i=0; i<NoOfTiles.x; i++)
        {
            for(int j=0; j<NoOfTiles.y; j++)
            {
//                if(!m_camera_changed)
                {
                    m_context["NoOfTiles"]->setUint(i, j);
	
					/// @todo Check for optix exception for timeouts etc
                    m_context->launch( static_cast<unsigned int>( m_cameraMode ),
                                      static_cast<unsigned int>(launch_index_tileSize.x),
                                      static_cast<unsigned int>(launch_index_tileSize.y)
                                      );

//                    updateGLBuffer();
//                    emit bucketReady(i, j);
                }
            }
//            updateGLBuffer();
//            emit bucketRowReady(i);
        }
        qDebug( "%d/%d iterations completed", m_frame - 1, m_progressiveTimeout - 1);
//        emit frameRefined(m_frame - 1);
    }
    else if(!m_frameDone) // We've hit the 'max' timeout
    {
        m_frameDone = true;
        isFrameReady = true;
    }

    updateGLBuffer();

    if(isFrameReady)
    {
        emit frameReady();
    }

    /// ===========================================================

  //  RT_CHECK_ERROR( sutilDisplayFilePPM( "/home/tom/src/OptixQt/out.ppm", buffer->get() ) );
}
