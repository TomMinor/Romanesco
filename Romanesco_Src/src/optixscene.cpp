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
#include <unistd.h>
#include <math.h>
#include <QtMath>
#include <QDir>
#include <fstream>
#include <string>
#include <cerrno>
#include <assert.h>

#include <OpenEXR/ImfRgba.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/half.h>

#include <iostream>
#include <boost/algorithm/string/join.hpp>

#include "optixscene.h"
#include <ImageLoader.h>
#include "runtimecompiler.h"


struct Image
{
public:
    Image( float* _pixels, unsigned int _width, unsigned int _height, std::string _name = "" )
        : m_width(_width), m_height(_height), m_name(_name)
  {
    m_pixels = new Imf::Rgba[m_width * m_height];
    std::fill(m_pixels, m_pixels + (m_width * m_height), Imf::Rgba(1.f, 1.f, 1.f, 1.f) );

    for(int i = 0; i < 4 * m_width * m_height; i+=4)
    {
        //unsigned int idx = i + (j * m_width);

        float R = _pixels[i];
        float G = _pixels[i + 1];
        float B = _pixels[i + 2];
        float A = _pixels[i + 3];

        //setPixel(i, j, Imf::Rgba(R, G, B, A) );
        m_pixels[i / 4] = Imf::Rgba(R, G, B, A);
    }
  }

  void setPixel(int x, int y, Imf::Rgba _val)
  {
    m_pixels[x + (y * m_width)] = _val;
  }

  ~Image()
  {
        //delete m_pixels;
  }

//private:
  Imf::Rgba* m_pixels;
  unsigned int m_width, m_height;
    std::string m_name;
};


std::string layerChannelString( std::string _layerName, std::string _channel )
{
    return (_layerName.size() == 0) ? _channel : _layerName + "." + _channel;
}

void writeRGBA2(std::string fileName, std::vector<Image> _layers)
{
    Imf::Header header(_layers[0].m_width, _layers[0].m_height);

    Imf::ChannelList& channels = header.channels();
    Imf::FrameBuffer framebuffer;

    for(unsigned int i = 0; i < _layers.size(); i++)
    {
        Image& _image = _layers[i];

        std::string name_r = layerChannelString(_image.m_name, "R");
        std::string name_g = layerChannelString(_image.m_name, "G");
        std::string name_b = layerChannelString(_image.m_name, "B");
        std::string name_a = layerChannelString(_image.m_name, "A");

        channels.insert( name_r, Imf::Channel(Imf::HALF) );
        channels.insert( name_g, Imf::Channel(Imf::HALF) );
        channels.insert( name_b, Imf::Channel(Imf::HALF) );
        channels.insert( name_a, Imf::Channel(Imf::HALF) );

        char* channel_rPtr = (char*) &(_image.m_pixels[0].r);
        char* channel_gPtr = (char*) &(_image.m_pixels[0].g);
        char* channel_bPtr = (char*) &(_image.m_pixels[0].b);
        char* channel_aPtr = (char*) &(_image.m_pixels[0].a);

        unsigned int xstride = sizeof( half ) * 4;
        unsigned int ystride = sizeof( half ) * 4 * _image.m_width;

        framebuffer.insert( name_r, Imf::Slice( Imf::HALF, channel_rPtr, xstride, ystride ) );
        framebuffer.insert( name_g, Imf::Slice( Imf::HALF, channel_gPtr, xstride, ystride ) );
        framebuffer.insert( name_b, Imf::Slice( Imf::HALF, channel_bPtr, xstride, ystride ) );
        framebuffer.insert( name_a, Imf::Slice( Imf::HALF, channel_aPtr, xstride, ystride ) );
    }

    Imf::OutputFile file(fileName.c_str(), header);
    file.setFrameBuffer( framebuffer );
    file.writePixels( _layers[0].m_height );
}















static const unsigned int WIDTH = 1280;
static const unsigned int HEIGHT = 720;

enum AnimState {
  ANIM_ALL,       // full auto
  ANIM_JULIA,     // manual camera
  ANIM_NONE       // pause all
};
static AnimState animstate = ANIM_ALL;


// Encapulates the state of the floor
struct FloorState
{
  FloorState()
    : m_t( 0 )
  {}

  void update( double t )
  {
    if( animstate==ANIM_NONE )
      return;
    m_t += t * 0.3f;
  }

  double m_t;
};

// Moving force particle.
struct Particle
{
  Particle()
    : m_pos( optix::make_float3(1,1,0) )
    , m_t( 0 )
  {}

  void update( double t )
  {
    if( animstate==ANIM_NONE )
      return;
    m_t += t * 0.3f;
    m_pos.x = (float)( sin( m_t ) * cos( m_t*0.2 ) );
    m_pos.y = (float)sin( m_t*2.5 );
    m_pos.z = (float)( cos( m_t*1.8 ) * sin( m_t ) );
  }

  optix::float3 m_pos;
  double m_t;
};

// Animated parameter quaternion.
static const int nposes = 12;
static const optix::float4 poses[nposes] = {
  {-0.5f, 0.1f, 0.2f, 0.3f },
  {-0.71f, 0.31f, -0.02f, 0.03f },
  {-0.5f, 0.1f, 0.59f, 0.03f },
  { -0.5f, -0.62f, 0.2f, 0.3f },
  {-0.57f, 0.04f, -0.17f, 0.36f },
  {0.0899998f, -0.71f, -0.02f, 0.08f },
  {-0.19f, -0.22f, -0.79f, 0.03f },
  {0.49f, 0.48f, -0.38f, -0.11f },
  {-0.19f, 0.04f, 0.0299999f, 0.77f },
  { 0.0299998f, -1.1f, -0.03f, -0.1f },
  {0.45f, 0.04f, 0.56f, -0.00999998f },
  { -0.5f, -0.61f, -0.08f, -0.00999998f }
};
struct ParamQuat
{
  ParamQuat()
    : m_c( optix::make_float4( -0.5f, 0.1f, 0.2f, 0.3f ) )
    , m_t( 0 )
  {}

  void update( double t )
  {
    if( animstate==ANIM_NONE )
      return;
    m_t += t * 0.03f;
    const float rem   = fmodf( (float)m_t, (float)nposes );
    const int   p0    = (int)rem;
    const int   p1    = (p0+1) % nposes;
    const float lin   = rem - (float)p0;
    const float blend = optix::smoothstep( 0.0f, 1.0f, lin );
    m_c = optix::lerp( poses[p0], poses[p1], blend );
  }

  optix::float4 m_c;
  double m_t;
};

// Animated camera.
struct AnimCamera
{
  AnimCamera()
    : m_pos( optix::make_float3(0) )
    , m_aspect( (float)WIDTH/(float)HEIGHT )
    , m_t( 0 )
  {}

  void update( double t )
  {
    m_t += t * 0.1;
    m_pos.y = (float)( 2 + sin( m_t*1.5 ) );
    m_pos.x = (float)( 2.3*sin( m_t ) );
    m_pos.z = (float)( 0.5+2.1*cos( m_t ) );
  }

  void apply( optix::Context context )
  {
    MyPinholeCamera pc( m_pos, optix::make_float3(0), optix::make_float3(0,1,0), 60.f, 60.f/m_aspect );
    optix::float3 eye, u, v, w;
    pc.getEyeUVW( eye, u, v, w );
    context["eye"]->setFloat( eye );
    context["U"]->setFloat( u );
    context["V"]->setFloat( v );
    context["W"]->setFloat( w );
  }

  optix::float3 m_pos;
  float  m_aspect;
  double m_t;
};

optix::Buffer OptixScene::createOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height)
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

OptixScene::OptixScene(unsigned int _width, unsigned int _height)
    : m_time(0.0f)
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
    m_context->setRayTypeCount( 2 );
    m_context->setEntryPointCount( 1 );
    m_context->setStackSize(1280);

    m_context["max_depth"]->setInt( 5 );
    m_context["radiance_ray_type"]->setUint( 0u );
    m_context["shadow_ray_type"]->setUint( 1u );
    m_context["scene_epsilon"]->setFloat( 1.e-4f );
    m_context["color_t"]->setFloat( 0.0f );
    m_context["shadowsActive"]->setUint( 0u );
    m_context["global_t"]->setFloat( 0u );

    updateBufferSize(_width, _height);

    camera_data = InitialCameraData( optix::make_float3( 3.0f, 2.0f, -3.0f ), // eye
                                     optix::make_float3( 0.0f, 0.3f,  0.0f ), // lookat
                                     optix::make_float3( 0.0f, 1.0f,  0.0f ), // up
                                     60.0f );                          // vfov

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );

//    setCamera( optix::make_float3( 3.0f, 2.0f, -3.0f ), // eye
//               optix::make_float3( 0.0f, 0.3f,  0.0f ), // lookat
//               60.f // fov
//               );

    //sprintf( path_to_ptx, "%s/%s", "ptx", "draw.cu.ptx" );
//    std::string ptx_path_str(  );
    optix::Program ray_gen_program = m_context->createProgramFromPTXFile( "ptx/pinhole_camera.cu.ptx", "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception
    optix::Program exception_program = m_context->createProgramFromPTXFile( "ptx/pinhole_camera.cu.ptx", "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 1.0f, 1.0f, 0.0f );

    // Miss program
    //m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( "ptx/constantbg.cu.ptx", "miss" ) );
    m_context["bg_color"]->setFloat( optix::make_float3(108.0f/255.0f, 166.0f/255.0f, 205.0f/255.0f) * 0.5f );

    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( "ptx/raymarch.cu.ptx", "envmap_miss" ) );

    const optix::float3 default_color = optix::make_float3(1.0f, 1.0f, 1.0f);
    //m_context["envmap"]->setTextureSampler( loadTexture( m_context, "/home/tom/src/optix/SDK/tutorial/data/CedarCity.hdr", default_color) );
//    m_context["envmap"]->setTextureSampler( loadTexture( m_context, "/home/tom/src/Fragmentarium/Fragmentarium-Source/Examples/Include/Ditch-River_2k.hdr", default_color) );
    m_context["envmap"]->setTextureSampler( loadTexture( m_context, "/home/tom/Downloads/Milkyway/Milkyway_small.hdr", default_color) );


    // Setup lights
    m_context["ambient_light_color"]->setFloat(0.1f,0.1f,0.3f);
    BasicLight lights[] = {
      { { 0.0f, 8.0f, -5.0f }, { .8f, .8f, .6f }, 1 },
    };

    optix::Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

    // Create scene geom
    createGeometry();

    m_camera = new MyPinholeCamera( camera_data.eye,
                                  camera_data.lookat,
                                  camera_data.up,
                                  -1.0f, // hfov is ignored when using keep vertical
                                  camera_data.vfov,
                                  MyPinholeCamera::KeepVertical );

//    setCamera( camera_data.eye,
//               camera_data.lookat,
//               60.0f,
//               _width, _height);



    //ray_gen_program["draw_color"]->setFloat( optix::make_float3(0.462f, 0.725f, 0.0f) );

    m_context->validate();
    m_context->compile();
}

void OptixScene::setCamera(optix::float3 _eye, /*optix::float3 _lookat, */float _fov, int _width, int _height)
{
    m_camera->setParameters( _eye,
                             optix::make_float3(0,0,0),
                             camera_data.up,
                             _fov, // hfov is ignored when using keep vertical
                             _fov,
                             MyPinholeCamera::KeepHorizontal );

    optix::float3 eye, U, V, W;
    m_camera->setAspectRatio( static_cast<float>(_width)/static_cast<float>(_height) );

    m_context["eye"]->setFloat( _eye );
    m_context["U"]->setFloat( optix::make_float3(1, 0, 0) );
    m_context["V"]->setFloat( optix::make_float3(0, 1, 0) );
    m_context["W"]->setFloat( optix::make_float3(0, 0, 1) );

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
    static std::vector<std::string> bufferNames = { "output_buffer",
                                                    "output_buffer_nrm",
                                                    "output_buffer_depth",
                                                    "output_buffer_world" };

    for( auto& bufferName : bufferNames )
    {

        if(m_context[bufferName]->getType() == RT_OBJECTTYPE_UNKNOWN )
        {
            m_context[bufferName]->set( createOutputBuffer(RT_FORMAT_FLOAT4, _width, _height) );
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
}

std::string get_file_contents(const std::string& _filename)
{
  std::ifstream in(_filename.c_str(), std::ios::in | std::ios::binary);
  if (in)
  {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return(contents);
  }
  throw(errno);
}

#include <sstream>
#include <algorithm>

bool hookPtxFunction( const std::string& _ptxPath,
                      const std::string& _functionName,
                      const std::string& _functionSource,
                      std::string& _result)
{
    std::string ptx = get_file_contents(_ptxPath);

    RuntimeCompiler program(_functionName, _functionSource);

    // Convert the function ptx to a vector of lines
    std::vector<std::string> src_ptx;
    {
        std::istringstream src_stream( program.getResult() );
        std::string line;
        while(std::getline(src_stream, line))
        {
            src_ptx.push_back(line);
        }
    }

    // Remove the compiler comments/version info
    // @todo Make this more robust
    src_ptx.erase(src_ptx.begin(), src_ptx.begin() + 13);

    // Convert the ptx code to a vector of lines
    std::vector<std::string> ptx_lines;
    // The start/end lines of the extern function block (@todo: these should be iterators)
    unsigned int startExtern = -1;
    unsigned int endExtern = -1;
    {
        std::istringstream ptx_stream(ptx);
        std::string line;

        bool foundExtern = false;
        bool foundExternEnd = false;
        unsigned int lineNum = 0;
        // Convert the ptx to a vector of lines and
        // check for the start/end of the extern block
        // of the function we're replacing
        while (std::getline(ptx_stream, line))
        {
            if( !foundExtern )
            {
                size_t externPos = line.find(".extern");
                size_t shade_hookPos = line.find(_functionName);

                if( externPos != std::string::npos && shade_hookPos != std::string::npos)
                {
                    foundExtern = true;
                    startExtern = lineNum;
                }
            }
            else if( !foundExternEnd )
            {
                size_t semicolonPos = line.find(";");
                if( semicolonPos != std::string::npos )
                {
                    endExtern = lineNum;
                    foundExternEnd = true;
                }
            }

            ptx_lines.push_back(line);
            lineNum++;
        }
    }

    // Remove old extern block
    ptx_lines.erase(ptx_lines.begin() + startExtern, ptx_lines.begin() + endExtern + 1);
    // Replace with our actual function ptx
    ptx_lines.insert(ptx_lines.begin() + startExtern, src_ptx.begin(), src_ptx.end());

    // Join the list into a string
    std::string concatptx = boost::algorithm::join(ptx_lines, "\n");

    // Return the string result
    _result = concatptx;

    return true;
}

#include "Base_SDFOP.h"
#include "Sphere_SDFOP.h"
#include "Transform_SDFOP.h"

void OptixScene::createGeometry(int choose)
{
    std::vector<BaseSDFOP*> ops;

    ops.push_back( new Sphere_SDFOP );
    ops.push_back( new Transform_SDFOP );

//    std::string shade_hook_src_A = ""
//    "#include \"cutil_math.h\" \n"
//    "extern \"C\" { \n"
//    "__device__ float3 shade_hook("
//            "float3 p, float3 nrm, float iteration"
//            ")"
//    "{\n"
//"        return nrm;\n"
//    "}\n } \n";

//    std::string shade_hook_src_B = ""
//    "#include \"cutil_math.h\" \n"
//    "extern \"C\" { \n"
//    "__device__ float3 shade_hook("
//            "float3 p, float3 nrm, float iteration"
//            ")"
//    "{\n"
//"        return p;\n"
//    "}\n } \n";

//    std::string shade_hook_src = shade_hook_src_B;//(choose == 0) ? shade_hook_src_A : shade_hook_src_B;


    std::string mandelbulb_hit_src =
            "#include \"cutil_math.h\" \n"

            "extern \"C\" {\n "
            "__device__ float distancehit_hook("
                    "float3 x, float _t, float _max_iterations"
                    ")\n"
            "{"
                "float3 zn  = x;//make_float3( x, 0 );\n"
                "float4 fp_n = make_float4( 1, 0, 0, 0 );  // start derivative at real 1 (see [2]).\n"
"\n"
                "const float sq_threshold = 2.0f;   // divergence threshold\n"
 "\n"
                "float oscillatingTime = sin(_t / 256.0f );\n"
                "float p = (5.0f * abs(oscillatingTime)) + 3.0f; //8;\n"
                "float rad = 0.0f;\n"
                "float dist = 0.0f;\n"
                "float d = 1.0;\n"
"\n"
                "// Iterate to compute f_n and fp_n for the distance estimator.\n"
                "int i = _max_iterations;\n"
                "while( i-- )\n"
                "{\n"
                  "rad = length(zn);\n"
            "\n"
                  "if( rad > sq_threshold )\n"
                  "{\n"
                    "dist = 0.5f * rad * logf( rad ) / d;\n"
                  "}\n"
                  "else\n"
                  "{\n"
                    "float th = atan2( length( make_float3(zn.x, zn.y, 0.0f) ), zn.z );\n"
                    "float phi = atan2( zn.y, zn.x );\n"
                    "float rado = pow(rad, p);\n"
                    "d = pow(rad, p - 1) * (p-1) * d + 1.0;\n"
"\n"
                    "float sint = sin(th * p);\n"
                    "zn.x = rado * sint * cos(phi * p);\n"
                    "zn.y = rado * sint * sin(phi * p);\n"
                    "zn.z = rado * cos(th * p);\n"
                    "zn += x;\n"
                  "}\n"
                "}\n"
            "\n"
                "return dist;\n"
            "}\n"
            "}\n";

    std::string sphere_hit_src =
            "#include \"cutil_math.h\" \n"
            "extern \"C\" {\n "
            "__device__ float distancehit_hook("
                    "float3 x, float _t, float _max_iterations"
                    ")\n"
            "{"
                "return length(x) - (sin(_t / 60.0f) + 1.0f);\n"
            "}\n"
            "}\n";

    std::string hit_src = (choose == 0) ? mandelbulb_hit_src : sphere_hit_src;

    std::string ptx;
    //hookPtxFunction("ptx/raymarch.cu.ptx", "shade_hook", shade_hook_src, ptx);

    qDebug() << mandelbulb_hit_src.c_str();

    hookPtxFunction("ptx/raymarch.cu.ptx", "distancehit_hook", hit_src, ptx);

    qDebug() << ptx.c_str();

    //qDebug() << result.c_str();

    //qDebug() << ptx.find("shade_hook");

//    qDebug() << ptx.c_str();

//    qDebug() << ptx.c_str();

    ///@todo Optix error checking

    optix::Geometry julia = m_context->createGeometry();
    julia->setPrimitiveCount( 1u );
    julia->setBoundingBoxProgram( m_context->createProgramFromPTXString( ptx, "bounds" ) );
    julia->setIntersectionProgram( m_context->createProgramFromPTXString( ptx, "intersect" ) );

    // Sphere
//    optix::Geometry sphere = m_context->createGeometry();
//    sphere->setPrimitiveCount( 1 );
//    sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( "ptx/sphere.cu.ptx", "bounds" ) );
//    sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( "ptx/sphere.cu.ptx", "intersect" ) );
//    m_context["sphere"]->setFloat( 1, 1, 1, 0.2f );

    optix::Program julia_ch = m_context->createProgramFromPTXString( ptx, "julia_ch_radiance" );
    optix::Program julia_ah = m_context->createProgramFromPTXString( ptx, "julia_ah_shadow" );
//    optix::Program chrome_ch = m_context->createProgramFromPTXString( ptx, "chrome_ch_radiance" );
//    optix::Program chrome_ah = m_context->createProgramFromPTXString( ptx, "chrome_ah_shadow" );
    //optix::Program floor_ch = m_context->createProgramFromPTXFile( "ptx/block_floor.cu.ptx", "block_floor_ch_radiance" );
    //optix::Program floor_ah = m_context->createProgramFromPTXFile( "ptx/block_floor.cu.ptx", "block_floor_ah_shadow" );
    //optix::Program normal_ch = m_context->createProgramFromPTXFile( "ptx/normal_shader.cu.ptx", "closest_hit_radiance" );

    // Julia material
    optix::Material julia_matl = m_context->createMaterial();
    julia_matl->setClosestHitProgram( 0, julia_ch );
    julia_matl->setAnyHitProgram( 1, julia_ah );

    // Sphere material
//    optix::Material sphere_matl = m_context->createMaterial();
//    sphere_matl->setClosestHitProgram( 0, chrome_ch );
//    sphere_matl->setAnyHitProgram( 1, chrome_ah );

//    m_context["Ka"]->setFloat(0.3f,0.3f,0.3f);
//    m_context["Kd"]->setFloat(.6f, 0.1f, 0.1f);
//    m_context["Ks"]->setFloat(.6f, .6f, .6f);
    m_context["Ka"]->setFloat(0.5f,0.0f,0.0f);
    m_context["Kd"]->setFloat(.6f, 0.1f, 0.1f);
    m_context["Ks"]->setFloat(.6f, .2f, .1f);
    m_context["phong_exp"]->setFloat(32);
    m_context["reflectivity"]->setFloat(.4f, .4f, .4f);

    // Place geometry into hierarchy
//    std::vector<optix::GeometryInstance> gis;
//    //gis.push_back( m_context->createGeometryInstance( sphere, &sphere_matl, &sphere_matl+1 ) );
//    gis.push_back( m_context->createGeometryInstance( julia,  &julia_matl, &julia_matl+1 ) );

//    m_geometrygroup = m_context->createGeometryGroup();
//    m_geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
//    for(size_t i = 0; i < gis.size(); ++i) {
//      m_geometrygroup->setChild( (int)i, gis[i] );
//    }

    std::vector<optix::GeometryInstance> gis;
    gis.push_back( m_context->createGeometryInstance( julia,  &julia_matl, &julia_matl+1 ) );

    m_geometrygroup = m_context->createGeometryGroup();
    m_geometrygroup->setChildCount( 1 );
    m_geometrygroup->setChild( (int)0, m_context->createGeometryInstance( julia,  &julia_matl, &julia_matl+1 ));
    m_geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

    // Top level group
    optix::Group topgroup = m_context->createGroup();
    topgroup->setChildCount( 1 );
    topgroup->setChild( 0, m_geometrygroup );
    topgroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

    m_context["top_object"]->set( topgroup );
    m_context["top_shadower"]->set( m_geometrygroup );

    float  m_alpha;
    float  m_delta;
    float m_DEL;
    unsigned int m_max_iterations;

    m_alpha = 0.003f;
    m_delta = 0.01f;
    m_DEL = 0.02f;
    m_max_iterations = 32;

    m_context[ "c4" ]->setFloat( optix::make_float4( -0.5f, 0.1f, 0.2f, 0.3f) );
    m_context[ "alpha" ]->setFloat( m_alpha );
    m_context[ "delta" ]->setFloat( m_delta );
    m_context[ "max_iterations" ]->setUint( m_max_iterations );
    m_context[ "DEL" ]->setFloat( m_DEL );
    m_context[ "particle" ]->setFloat( 0.5f, 0.5f, 0.4f );
    m_context[ "global_t" ]->setFloat( m_time );

    // set floor parameters
    //m_context[ "floor_time" ]->setFloat( (float)m_floorstate.m_t );
}

OptixScene::~OptixScene()
{

}

void OptixScene::drawToBuffer()
{
    RTsize buffer_width, buffer_height;
    m_context["output_buffer"]->getBuffer()->getSize( buffer_width, buffer_height );
    m_context->launch( 0,
                       static_cast<unsigned int>(buffer_width),
                       static_cast<unsigned int>(buffer_height)
                       );


    /// ==================  Copy to texture =======================

    optix::Buffer buffer = m_context["output_buffer"]->getBuffer();
    RTformat buffer_format = buffer->getFormat();



    // Debug dump
//    {
//        const unsigned int totalPixels = 4 * static_cast<unsigned int>(buffer_width) * static_cast<unsigned int>(buffer_height);

//        float* h_ptrDiffuse = new float[totalPixels];
//        CUdeviceptr d_ptrDiffuse = buffer->getDevicePointer( 0 );
//        cudaMemcpy( (void*)h_ptrDiffuse,   (void*)d_ptrDiffuse,    sizeof(float) * totalPixels, cudaMemcpyDeviceToHost );

//        std::vector<Image> passes;

//        passes.push_back( Image( h_ptrDiffuse, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height)) );

//        writeRGBA2("test.exr", passes );
//    }


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

        // Initialize offsets to pixel center sampling.

  //      float u = 0.5f/buffer_width;
  //      float v = 0.5f/buffer_height;
    }

    /// ===========================================================

  //  RT_CHECK_ERROR( sutilDisplayFilePPM( "/home/tom/src/OptixQt/out.ppm", buffer->get() ) );
}
