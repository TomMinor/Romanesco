#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <QWindow>
#include <QOpenGLFunctions>
#include <QOpenGLPaintDevice>
#include <QOpenGLFramebufferObject>
#include <QScreen>
#include <QDebug>
#include <QKeyEvent>
#include <QtMath>
#include <QDir>

#include <unistd.h>
#include <math.h>
#include <fstream>
#include <string>
#include <cerrno>
#include <assert.h>
#include <iostream>

#include <boost/algorithm/string/join.hpp>

#include <ImageLoader.h>

#include "stringutilities.h"
#include "ImageWriter.h"
#include "OptixScene.h"
#include "RuntimeCompiler.h"

#define USE_DEBUG_EXCEPTIONS 1

#include <algorithm>
#include "Base_SDFOP.h"
#include "Primitive/Sphere_SDFOP.h"
#include "DomainOp/Transform_SDFOP.h"
#include <glm/gtc/matrix_transform.hpp>
#include "path_tracer/path_tracer.h"

///@todo
/// * Split this into a simple base class and derive from that, OptixScene -> OptixSceneAdaptive -> OptixScenePathTracer
/// * All camera stuff should be moved into it's own, simpler class
///
optix::Buffer OptixScene::createGLOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height)
{
    optix::Buffer buffer;

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    size_t element_size;
    m_context->checkError( rtuGetSizeForRTformat(_format, &element_size) );
    glBufferData(GL_ARRAY_BUFFER, element_size * _width * _height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

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

OptixScene::OptixScene(unsigned int _width, unsigned int _height, QObject *_parent)
    : QObject(_parent), m_time(0.0f)
{
    /// ================ Output Texture Buffer ======================

    glGenTextures( 1, &m_texId );
    glBindTexture( GL_TEXTURE_2D, m_texId);

    // Change these to GL_LINEAR for super- or sub-sampling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture( GL_TEXTURE_2D, 0);

    /// =============================================================

    m_context = optix::Context::create();
    m_context->setRayTypeCount( 3 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize( 1800 );

    updateBufferSize(_width, _height);

    // Create scene geom
    createGeometry();

//    m_camera = new MyPinholeCamera( camera_data.eye,
//                                  camera_data.lookat,
//                                  camera_data.up,
//                                  -1.0f, // hfov is ignored when using keep vertical
//                                  camera_data.vfov,
//                                  MyPinholeCamera::KeepVertical );
//
//    setCamera( camera_data.eye,
//               camera_data.lookat,
//               60.0f,
//               _width, _height);

    setOutputBuffer("output_buffer");

    //ray_gen_program["draw_color"]->setFloat( optix::make_float3(0.462f, 0.725f, 0.0f) );

#if USE_DEBUG_EXCEPTIONS
    // Disable this by default for performance, otherwise the stitched PTX code will have lots of exception handling inside.
    m_context->setPrintEnabled(true);
    m_context->setPrintLaunchIndex(256, 256); // Launch index (0,0) at lower left.
    m_context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
#endif

    m_context->validate();
    m_context->compile();

    m_progressiveTimeout = 20;
}

void OptixScene::setTime(float _t)
{
    m_time = (_t / 10.0f);
    m_context[ "global_t" ]->setFloat( m_time );
}

void OptixScene::setCamera(optix::float3 _eye, /*optix::float3 _lookat, */float _fov, int _width, int _height)
{
//    m_camera->setParameters( _eye,
//                             optix::make_float3(0,0,0),
//                             camera_data.up,
//                             _fov, // hfov is ignored when using keep vertical
//                             _fov,
//                             MyPinholeCamera::KeepHorizontal );

//    optix::float3 eye, U, V, W;
//    m_camera->setAspectRatio( static_cast<float>(_width)/static_cast<float>(_height) );

    m_context["eye"]->setFloat( _eye );
    m_context["U"]->setFloat( optix::make_float3(1, 0, 0) );
    m_context["V"]->setFloat( optix::make_float3(0, 1, 0) );
    m_context["W"]->setFloat( optix::make_float3(0, 0, 1) );

    m_camera_changed = true;

//    m_camera->getEyeUVW( eye, U, V, W );

//    m_context["eye"]->setFloat( eye );
//    m_context["U"]->setFloat( U );
//    m_context["V"]->setFloat( V );
//    m_context["W"]->setFloat( W );
}

void OptixScene::setVar(const std::string& _name, float _v)
{
    m_context[_name]->setFloat(_v);
}

void OptixScene::setOutputBuffer(std::string _name)
{
    m_outputBuffer = _name;
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
    static std::vector<std::pair<std::string, RTformat>> glOutputBuffers = {
        std::make_pair("output_buffer",         RT_FORMAT_FLOAT4),
        std::make_pair("output_buffer_nrm",     RT_FORMAT_FLOAT3),
        std::make_pair("output_buffer_world",   RT_FORMAT_FLOAT3),
        std::make_pair("output_buffer_depth",   RT_FORMAT_FLOAT)
    };

    static std::vector<std::pair<std::string, RTformat>> outputBuffers = {

    };

    // Update any GL bound Optix buffers
    for( auto& buffer : glOutputBuffers )
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
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_context[bufferName]->getBuffer()->getGLBOId());
            glBufferData(GL_PIXEL_UNPACK_BUFFER, m_context[bufferName]->getBuffer()->getElementSize() * _width * _height, 0, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            m_context[bufferName]->getBuffer()->registerGLBuffer();
        }
    }

    // Update regular Optix buffers
    for( auto& buffer : outputBuffers )
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


GeometryInstance createAreaLight( optix::Context* m_context,
                                      optix::Program* m_pgram_bounding_box,
                                      optix::Program* m_pgram_intersection,
                                      const float3& anchor,
                                      const float3& offset1,
                                      const float3& offset2)
{
  Geometry parallelogram = (*m_context)->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( *m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( *m_pgram_bounding_box );

  float3 normal = normalize( cross( offset1, offset2 ) );
  float d = dot( normal, anchor );
  float4 plane = make_float4( normal, d );

  float3 v1 = offset1 / dot( offset1, offset1 );
  float3 v2 = offset2 / dot( offset2, offset2 );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );

  GeometryInstance gi = (*m_context)->createGeometryInstance();
  gi->setGeometry(parallelogram);
  return gi;
}

void setMaterial( GeometryInstance& gi,
                                   Material material,
                                   const std::string& color_name,
                                   const float3& color)
{
  gi->addMaterial(material);
  gi[color_name]->setFloat(color);
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

    std::string hit_src = (_hit_src == "") ? geometryhook_src : _hit_src;

    // Compile function source to ptx
    RuntimeCompiler program("distancehit_hook", _hit_src);
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
    } catch(Exception &e)
    {
        qWarning("Failed to create program from ptx");
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
    } catch(Exception &e)
    {
        qWarning("Failed to create program from ptx");
        return;
    }
}

void OptixScene::createGeometry()
{
    m_context["max_depth"]->setInt( 5 );
//    m_context["radiance_ray_type"]->setUint( 0u );
//    m_context["shadow_ray_type"]->setUint( 1u );
    m_context["scene_epsilon"]->setFloat( 1.e-4f );
    m_context["color_t"]->setFloat( 0.0f );
    m_context["shadowsActive"]->setUint( 0u );
    m_context["global_t"]->setFloat( 0u );

    m_context["scene_epsilon"]->setFloat( 1.e-3f );
    m_context["pathtrace_ray_type"]->setUint(0u);
    m_context["pathtrace_shadow_ray_type"]->setUint(1u);
    m_context["pathtrace_bsdf_shadow_ray_type"]->setUint(2u);
    m_context["rr_begin_depth"]->setUint(m_rr_begin_depth);

//    camera_data = InitialCameraData( optix::make_float3( 3.0f, 2.0f, -3.0f ), // eye
//                                     optix::make_float3( 0.0f, 0.3f,  0.0f ), // lookat
//                                     optix::make_float3( 0.0f, 1.0f,  0.0f ), // up
//                                     60.0f );                          // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );


    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

    // Miss program
    //m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( "ptx/constantbg.cu.ptx", "miss" ) );
    m_context["bg_color"]->setFloat( optix::make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( "ptx/raymarch.cu.ptx", "envmap_miss" ) );

    const optix::float3 default_color = optix::make_float3(1.0f, 1.0f, 1.0f);
//    m_context["envmap"]->setTextureSampler( loadTexture( m_context, "/home/i7245143/src/optix/SDK/tutorial/data/CedarCity.hdr", default_color) );
//    m_context["envmap"]->setTextureSampler( loadTexture( m_context, "/home/tom/src/Fragmentarium/Fragmentarium-Source/Examples/Include/Ditch-River_2k.hdr", default_color) );
    m_context["envmap"]->setTextureSampler( loadTexture( m_context,  qgetenv("HOME").toStdString() + "/Downloads/Milkyway/Milkyway_small.hdr", default_color) );


    m_rr_begin_depth = 1u;
    m_sqrt_num_samples = 1u;
    m_camera_changed = true;


    // Setup path tracer
    m_context["sqrt_num_samples"]->setUint( m_sqrt_num_samples );
    m_context["frame_number"]->setUint(1);

    // Index of sampling_stategy (BSDF, light, MIS)
    m_sampling_strategy = 0;
    m_context["sampling_stategy"]->setInt(m_sampling_strategy);

    // Setup lights
    m_context["ambient_light_color"]->setFloat(0.1f,0.1f,0.3f);

    float3 test_data[] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f }
    };

    optix::Buffer test_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, sizeof(test_data)/sizeof(test_data[0]) );
    memcpy( test_buffer->map(), test_data, sizeof(test_data) );
    test_buffer->unmap();

    m_context["test"]->set(test_buffer);

    optix::Program ray_gen_program = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "pathtrace_camera" );
    optix::Program exception_program = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "exception" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );
    m_context->setExceptionProgram( 0, exception_program );

    ///@todo Optix error checking
    optix::Geometry julia = m_context->createGeometry();
    julia->setPrimitiveCount( 1u );

    julia->setBoundingBoxProgram( m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "bounds" ) );
    julia->setIntersectionProgram( m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "intersect" ) );

    m_context["Ka"]->setFloat(0.5f,0.0f,0.0f);
    m_context["Kd"]->setFloat(.6f, 0.1f, 0.1f);
    m_context["Ks"]->setFloat(.6f, .2f, .1f);
    m_context["phong_exp"]->setFloat(32);
    m_context["reflectivity"]->setFloat(.4f, .4f, .4f);

    ParallelogramLight light;
    light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
    light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
    light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
    light.normal   = normalize( cross(light.v1, light.v2) );
    light.emission = make_float3( 100.0f );

    Buffer light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( 1u );
    memcpy( light_buffer->map(), &light, sizeof( light ) );
    light_buffer->unmap();
    m_context["lights"]->setBuffer( light_buffer );

    std::vector<optix::GeometryInstance> gis;

    GeometryInstance gi = m_context->createGeometryInstance();
    gi->setGeometry(julia);

    //@todo Critical : Fix this :|
    std::string ptx_path = qgetenv("OPTIX_PATH").toStdString() + "/build/lib/ptx/path_tracer_generated_parallelogram.cu.ptx";
    auto m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
    auto m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect" );

    const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
    const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
    const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
    const float3 light_em = make_float3( 15.0f, 15.0f, 5.0f );

    Material diffuse = m_context->createMaterial();
    Program diffuse_ch = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "diffuse" );
    Program diffuse_ah = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "shadow" );
    diffuse->setClosestHitProgram( 0, diffuse_ch );
    diffuse->setAnyHitProgram( 1, diffuse_ah );

    Material diffuse_light = m_context->createMaterial();
    Program diffuse_em = m_context->createProgramFromPTXFile( "ptx/menger.cu.ptx", "diffuseEmitter" );
    diffuse_light->setClosestHitProgram( 0, diffuse_em );

    gis.push_back( gi );
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Create shadow group (no light)
    GeometryGroup shadow_group = m_context->createGeometryGroup(gis.begin(), gis.end());
    shadow_group->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
//    m_context["top_shadower"]->set( shadow_group );

    // Light
    gis.push_back( createAreaLight( &m_context,
                                        &m_pgram_bounding_box,
                                        &m_pgram_intersection,
                                        make_float3( -2500, 2000.0, -2500),
                                        make_float3( 5000.0f, 0.0f, 0.0f),
                                        make_float3( 0.0f, 0.0f, 5000.0f) ) );
    setMaterial(gis.back(), diffuse_light, "emission_color", light_em);

    GeometryGroup m_geometrygroup = m_context->createGeometryGroup();
    m_geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
    for(size_t i = 0; i < gis.size(); ++i)
    {
        m_geometrygroup->setChild( (int)i, gis[i] );
    }
    m_geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

    // Top level group
    Group topgroup = m_context->createGroup();
    topgroup->setChildCount( 1 );
    topgroup->setChild( 0, m_geometrygroup );
    //topgroup->setChild( 1, floor_gg );
    topgroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );


    m_context["top_object"]->set( m_geometrygroup );
    m_context["top_shadower"]->set( m_geometrygroup );

    float  m_alpha;
    float  m_delta;
    float m_DEL;
    unsigned int m_max_iterations;

    m_alpha = 0.003f;
    m_delta = 0.00001f;
    m_DEL = 0.0001f;
    m_max_iterations = 32;

    m_context[ "c4" ]->setFloat( optix::make_float4( -0.5f, 0.1f, 0.2f, 0.3f) );
    m_context[ "alpha" ]->setFloat( m_alpha );
    m_context[ "delta" ]->setFloat( m_delta );
    m_context[ "max_iterations" ]->setUint( m_max_iterations );
    m_context[ "DEL" ]->setFloat( m_DEL );
    m_context[ "particle" ]->setFloat( 0.5f, 0.5f, 0.4f );
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
    cudaMemcpy( (void*)hostPtr,   (void*)devicePtr,    buffer->getElementSize() * bufferSize, cudaMemcpyDeviceToHost );

    return hostPtr;
}

float* OptixScene::getBufferContents(std::string _name)
{
    RTsize ignore, width, height;
    return getBufferContents(_name, &ignore, &width, &height);
}

std::string OptixScene::outputBuffer()
{
    return m_outputBuffer;
}

bool OptixScene::saveBuffersToDisk(std::string _filename)
{
    float* diffuse = getBufferContents("output_buffer");
    float* normal = getBufferContents("output_buffer_nrm");
    float* world = getBufferContents("output_buffer_world");
    float* depth = getBufferContents("output_buffer_depth");

    RTsize buffer_width, buffer_height;
    m_context["output_buffer"]->getBuffer()->getSize(buffer_width, buffer_height);
    const unsigned int totalPixels = static_cast<unsigned int>(buffer_width) * static_cast<unsigned int>(buffer_height);

    std::vector<ImageWriter::Pixel> pixels;
    pixels.reserve(totalPixels);

    ///@Todo move this into one loop, messing up my pointer arithmetic when i do so right now
    for(unsigned int i = 0; i < totalPixels * 4; i+=4)
    {
        ImageWriter::Pixel tmp;
        tmp.r = diffuse[i + 0];
        tmp.g = diffuse[i + 1];
        tmp.b = diffuse[i + 2];
        tmp.a = diffuse[i + 3];

        pixels.push_back(tmp);
    }

    for(unsigned int j = 0; j < totalPixels; j++)
    {
        pixels[j].x_nrm = normal[3*j + 0];
        pixels[j].y_nrm = normal[3*j + 1];
        pixels[j].z_nrm = normal[3*j + 2];

        pixels[j].x_pos = world[3*j + 0];
        pixels[j].y_pos = world[3*j + 1];
        pixels[j].z_pos = world[3*j + 2];
    }

    for(unsigned int k = 0; k < totalPixels; k++)
    {
        pixels[k].z = depth[k];
    }


    delete [] diffuse;
    delete [] normal;
    delete [] world;
    delete [] depth;

    diffuse = nullptr;
    normal = nullptr;
    world = nullptr;
    depth = nullptr;

    ImageWriter image(_filename, buffer_width, buffer_height);
    return image.write(pixels);
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

    // Update Optix scene if necessary
    if(m_frame < m_progressiveTimeout)
    {
        m_context["frame_number"]->setUint(m_frame);
        m_frame++;

        m_context->launch( 0,
                           static_cast<unsigned int>(buffer_width),
                           static_cast<unsigned int>(buffer_height)
                           );
    }
    else if(!m_frameDone) // We've hit the 'max' timeout
    {
        m_frameDone = true;
        emit frameReady();
    }

    // Copy optix buffer to gl texture directly on the GPU
    // (current visible buffer could potentially be changed even when optix scene is finished rendering, so copy over this every frame regardless)
    /// ==================  Copy to texture =======================
    optix::Buffer buffer = m_context[m_outputBuffer]->getBuffer();
    RTformat buffer_format = buffer->getFormat();

    vboId = buffer->getGLBOId();

    if (vboId)
    {
        glBindTexture( GL_TEXTURE_2D, m_texId );

        // send pbo to texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vboId);

        RTsize elementSize = buffer->getElementSize();
        if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        {
            if(buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
            } else if(buffer_format == RT_FORMAT_FLOAT4) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
            } else if(buffer_format == RT_FORMAT_FLOAT3) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
            } else if(buffer_format == RT_FORMAT_FLOAT) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
            } else {
                assert(0 && "Unknown buffer format");
            }
        }

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    }
    else
    {
        assert(0 && "Couldn't bind GL Buffer Object");
    }


    /// ===========================================================

  //  RT_CHECK_ERROR( sutilDisplayFilePPM( "/home/tom/src/OptixQt/out.ppm", buffer->get() ) );
}
