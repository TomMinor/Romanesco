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

struct RayGenCameraData
  {
    RayGenCameraData() {}
    RayGenCameraData( const optix::float3& m_eye, const optix::float3& m_U, const optix::float3& m_V, const optix::float3& m_W )
      : eye(m_eye), U(m_U), V(m_V), W(m_W) {}
    optix::float3 eye;
    optix::float3 U;
    optix::float3 V;
    optix::float3 W;
  };

struct InitialCameraData
{
  InitialCameraData() {}
  InitialCameraData( optix::float3 m_eye, optix::float3 m_lookat, optix::float3 m_up, float  m_vfov )
    : eye(m_eye), lookat(m_lookat), up(m_up), vfov(m_vfov) {}

  optix::float3 eye;
  optix::float3 lookat;
  optix::float3 up;
  float  vfov;
};

class OptixScene
{
public:
    OptixScene(unsigned int _width, unsigned int _height);
    ~OptixScene();

    void updateBufferSize(unsigned int _width, unsigned int _height);
    void drawToBuffer();
    void createGeometry(int choose = 0);

    optix::Buffer createOutputBuffer(RTformat _format, unsigned int _width, unsigned int _height);

    void setCamera(optix::float3 _eye, float _fov, int _width, int _height);
    void setVar(const std::string& _name, float _v);
    void setVar(const std::string& _name, optix::float3 _v);
    void setVar(const std::string& _name, optix::Matrix4x4 _v);

    void testSceneUpdate(int count)
    {
        optix::Program julia_ch = m_context->createProgramFromPTXFile( "ptx/julia.cu.ptx", "julia_ch_radiance" );
        optix::Program julia_ah = m_context->createProgramFromPTXFile( "ptx/julia.cu.ptx", "julia_ah_shadow" );
        optix::Program chrome_ch = m_context->createProgramFromPTXFile( "ptx/julia.cu.ptx", "chrome_ch_radiance" );
        optix::Program chrome_ah = m_context->createProgramFromPTXFile( "ptx/julia.cu.ptx", "chrome_ah_shadow" );

        optix::Geometry julia = m_context->createGeometry();
        julia->setPrimitiveCount( 1u );
        julia->setBoundingBoxProgram( m_context->createProgramFromPTXFile( "ptx/julia.cu.ptx", "bounds" ) );
        julia->setIntersectionProgram( m_context->createProgramFromPTXFile( "ptx/julia.cu.ptx", "intersect" ) );

        optix::Geometry sphere = m_context->createGeometry();
        sphere->setPrimitiveCount( 1 );
        sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( "ptx/sphere.cu.ptx", "bounds" ) );
        sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( "ptx/sphere.cu.ptx", "intersect" ) );

        // Julia material
        optix::Material julia_matl = m_context->createMaterial();
        julia_matl->setClosestHitProgram( 0, julia_ch );
        julia_matl->setAnyHitProgram( 1, julia_ah );

        // Sphere material
        optix::Material sphere_matl = m_context->createMaterial();
        sphere_matl->setClosestHitProgram( 0, chrome_ch );
        sphere_matl->setAnyHitProgram( 1, chrome_ah );


        std::vector<optix::GeometryInstance> gis;
        for(int i = 0; i < count; i++)
        {
            gis.push_back( m_context->createGeometryInstance( sphere, &sphere_matl, &sphere_matl+1 ) );
            m_context["sphere"]->setFloat( count, count, count, count * 0.2f );
            //gis.push_back( m_context->createGeometryInstance( julia,  &julia_matl, &julia_matl+1 ) );
        }

        m_geometrygroup = m_context->createGeometryGroup();
        m_geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
        for(size_t i = 0; i < gis.size(); ++i) {
          m_geometrygroup->setChild( (int)i, gis[i] );
        }
        m_geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

        // Top level group
        optix::Group topgroup = m_context->createGroup();
        topgroup->setChildCount( 1 );
        topgroup->setChild( 0, m_geometrygroup );
        topgroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

        m_context["top_object"]->set( topgroup );
        m_context["top_shadower"]->set( m_geometrygroup );
    }

    InitialCameraData camera_data;
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
};

#endif // OPTIXSCENE_H
