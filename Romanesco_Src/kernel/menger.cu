
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */


#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "distance_field.h"
#include "path_tracer.h"
#include "random.h"

#include "GLSL_Functions.h"

using namespace optix;

#define USE_DEBUG_EXCEPTIONS 1

// References:
// [1] Hart, J. C., Sandin, D. J., and Kauffman, L. H. 1989. Ray tracing deterministic 3D fractals
// [2] http://www.devmaster.net/forums/showthread.php?t=4448


rtDeclareVariable( float3, eye, , );
rtDeclareVariable( float4, c4 , , );                // parameter quaternion
rtDeclareVariable( float,  alpha , , );
rtDeclareVariable( float,  delta , , );
rtDeclareVariable( float,  DEL , , );
rtDeclareVariable( float,  color_t , , );           // 0,1,2 are full colors, in between is morph
rtDeclareVariable( uint,   max_iterations , , );    // max iterations for divergence determination
rtDeclareVariable( float3, particle , , );          // position of force particle
rtDeclareVariable( float3, center , , );          // position of force particle
rtDeclareVariable( float, global_t, , );          // position of force particle

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

// julia set object outputs this
rtDeclareVariable(float3, normal, attribute normal, );

// sphere outputs this
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, shading_normal2, attribute shading_normal2, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float3, 2>              output_buffer_nrm;
rtBuffer<float3, 2>              output_buffer_world;
rtBuffer<float, 2>              output_buffer_depth;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(Matrix4x4, normalmatrix, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );

rtDeclareVariable(rtObject,                         top_object, , );
rtDeclareVariable(rtObject,                         top_shadower, , );
rtDeclareVariable(float, isect_t, rtIntersectionDistance, );
//rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
//rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );

rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );

rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );

//struct PerRayData_radiance
//{
//  float3 result;
//  float3 result_nrm;
//  float3 result_world;
//  float result_depth;
//  int depth;

//  int iter;
//};

struct PerRayData_pathtrace
{
  float4 result;
  float3 result_nrm;
  float3 result_world;
  float result_depth;

  float3 origin;
  float3 radiance;
  float3 direction;
  float3 attenuation;
  unsigned int seed;
  int depth;
  int countEmitted;
  int done;
  int inside;

};

struct PerRayData_pathtrace_shadow
{
  bool inShadow;
};

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace, prd_radiance, rtPayload, );


//struct PerRayData_shadow
//{
//  float3 attenuation;
//  bool inShadow;
//};




RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_float4(bad_color, 0.0f);
  output_buffer_nrm[launch_index] = make_float3(0.0, 0.0, 0.0);
  output_buffer_world[launch_index] = make_float3(0.0, 0.0, 0.0);
  output_buffer_depth[launch_index] = RT_DEFAULT_MAX;

#if USE_DEBUG_EXCEPTIONS
  const unsigned int code = rtGetExceptionCode();
  rtPrintf("Exception code 0x%X at (%d, %d)\n", code, launch_index.x, launch_index.y);
#endif
}


RT_PROGRAM void pathtrace_camera()
{
    size_t2 screen = output_buffer.size();

    float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen / sqrt_num_samples;
    unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;

    // Store accumulated radiance, world position, normal and depth
    float4 result = make_float4(0.0f);
    float3 normal = make_float3(0.0f);
    float3 world = make_float3(0.0f);
    float depth = 0.0f;

    // Bounce GI
    unsigned int seed = tea<4>(screen.x * launch_index.y + launch_index.x, frame_number);
    do
    {
        unsigned int x = samples_per_pixel % sqrt_num_samples;
        unsigned int y = samples_per_pixel / sqrt_num_samples;
        float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
        float2 d = pixel + jitter*jitter_scale;
        float3 ray_origin = eye;
        float3 ray_direction = normalize(d.x*U + d.y*V + W);

        ray_direction = make_float3((make_float4(ray_direction, 1.0) * normalmatrix));
//        ray_direction = normalize(ray_direction);

        PerRayData_pathtrace prd;
        prd.result = make_float4(0.f);
        prd.result_nrm = make_float3(0.0f);
        prd.result_world = make_float3(0.0f);
        prd.result_depth = 0.0f;
        prd.attenuation = make_float3(1.0);
        prd.radiance = make_float3(0.0);
        prd.countEmitted = true;
        prd.done = false;
        prd.seed = seed;
        prd.depth = 0;

        Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray, prd);

//        prd.result_nrm.x = abs(prd.result_nrm.x);
//        prd.result_nrm.y = abs(prd.result_nrm.y);
//        prd.result_nrm.z = abs(prd.result_nrm.z);

        result += prd.result;
        normal += prd.result_nrm;
        world += prd.result_world;
        depth += prd.result_depth;

        seed = prd.seed;
    } while (--samples_per_pixel);

    float4 pixel_color = result/(sqrt_num_samples*sqrt_num_samples);
    float3 pixel_color_normal = normal/(sqrt_num_samples*sqrt_num_samples);
    float3 pixel_color_world = world/(sqrt_num_samples*sqrt_num_samples);
    float pixel_color_depth = depth/(sqrt_num_samples*sqrt_num_samples);

    // Smoothly blend with previous frames value
    if (frame_number > 1){
        float a = 1.0f / (float)frame_number;
        float b = ((float)frame_number - 1.0f) * a;

        float4 old_color = output_buffer[launch_index];
        output_buffer[launch_index] = a * pixel_color + b * old_color;

        float3 old_nrm = output_buffer_nrm[launch_index];
        output_buffer_nrm[launch_index] = a * pixel_color_normal + b * old_nrm;

        float3 old_world = output_buffer_world[launch_index];
        output_buffer_world[launch_index] = a * pixel_color_world + b * old_world;

        float old_depth = output_buffer_depth[launch_index];
        output_buffer_depth[launch_index] = a * pixel_color_depth + b * old_depth;
    }
    else
    {
        output_buffer[launch_index] = pixel_color;
        output_buffer_nrm[launch_index] = pixel_color_normal;
        output_buffer_world[launch_index] = pixel_color_world;
        output_buffer_depth[launch_index] = pixel_color_depth;
    }
}


//// Quaternion helpers.
//static __host__ __device__ float4 mul( float4 a, float4 b )
//{
//  const float3 a2 = make_float3( a.y, a.z, a.w );
//  const float3 b2 = make_float3( b.y, b.z, b.w );
//  float3 r;
//  r = a.x*b2 + b.x*a2 + cross( a2, b2 );
//  return make_float4(
//                      a.x*b.x - dot( a2, b2 ),
//                      r.x,
//                      r.y,
//                      r.z );
//}
//static __host__ __device__ float4 square( float4 a )
//{
//  float3 a2 = make_float3( a.y, a.z, a.w );
//  float3 r;
//  r = 2*a.x*a2;
//  return make_float4(
//    a.x*a.x - dot( a2,a2 ),
//    r.x, r.y, r.z );
//}

// Intersect the bounding sphere of the Julia set.
static __host__ __device__ bool intersectBoundingSphere( float3 o, float3 d, float sqRadius, float& tmin, float &tmax )
{
  const float sq_radius = sqRadius;
  const float b = dot( o, d );
  const float c = dot( o, o ) - sq_radius;
  const float disc = b*b - c;

  if( disc > 0.0f )
  {
    const float sdisc = sqrtf( disc );
    tmin = (-b - sdisc);
    tmax = (-b + sdisc);

    if(tmin > tmax)
    {
      const float temp = tmin;
      tmax = tmin;
      tmin = temp;
    }

    return true;
  }
  else
  {
    tmin = tmax = 0;
  }
  return false;
}



#define inf 10000.0

__device__ float sdCross(float3 _p)
{
    float3 p1 = make_float3(_p.x, _p.y, _p.z);
    float3 p2 = make_float3(_p.y, _p.z, _p.x);
    float3 p3 = make_float3(_p.z, _p.x, _p.y);

    float da = sdBox(p1, make_float3(inf, 1.0, 1.0));
    float db = sdBox(p2, make_float3(1.0, inf, 1.0));
    float dc = sdBox(p3, make_float3(1.0, 1.0, inf));

//    float da = maxcomp(fabs(p1));
//    float db = maxcomp(fabs(p2));
//    float dc = maxcomp(fabs(p3));

    return min(da, min(db, dc));
}

__device__ float2 rotate(float2 v, float a) {
    return make_float2( cos(a)*v.x + sin(a)*v.y, -sin(a)*v.x + cos(a)*v.y );
}


#define Iterations 32
#define Scale 2.0
#define Offset make_float3(0.92858,0.92858,0.32858)
#define FudgeFactor 0.7
__device__ float DE(float3 _p)
{
    float3 offset = make_float3(1.0 + 0.2f * cos(global_t / 5.7f),
                                1.0,
                                0.3 + 0.1f * (cos(global_t / 1.7f))
                                );

    float3 z = _p;



    z = fabs( 1.0 - fmod(z, 2.0));
    z.z = fabs(z.z + Offset.z) - Offset.z;

    float d = 1000.0f;
    for(int n = 0; n < Iterations; ++n)
    {
        ///@todo rotate

        // y
        if(z.x + z.y < 0.0){ float3 tmp = z; z.x = -tmp.y; z.y = -tmp.x; }
        z = fabs(z);

        // z
        if(z.x + z.z < 0.0){ float3 tmp = z; z.x = -tmp.z; z.z = -tmp.x; }
        z = fabs(z);

        // y
        if(z.x - z.y < 0.0){ float3 tmp = z; z.x = tmp.y; z.y = tmp.x; }
        z = fabs(z);

        // z
        if(z.x - z.z < 0.0){ float3 tmp = z; z.x = tmp.z; z.z = tmp.x; }
        z = fabs(z);

        z = z * Scale - offset * (Scale - 1.0);

        float2 tmp = make_float2(z.y, z.z);
        float2 r = rotate(tmp, -global_t / 18.0f);
        z.y = r.x;
        z.z = r.y;

        d = min(d, length(z) * powf(Scale, -float(n+1)));
    }

    return d;
}


__device__ float map(float3 _p)
{
    float a = DE(_p) * FudgeFactor;
    return a;

//    float d = sdBox(_p, make_float3(1.0f));

//    float s = 1.0;
//    for(int m=0; m<5; m++)
//    {
//        float3 a = fmod(_p * s, 2.0f) - make_float3(1.0f);
//        s *= 3.0;

//        float3 r = ( make_float3(1.0) - ( make_float3(3.0) * fabs(a)));

//        float c = (float)sdCross(r) / (float)s;

//        d = max(d, c);
//    }

//    return d;
}






struct JuliaSet
{
  __host__ __device__
  JuliaSet(const unsigned int max_iterations) : m_max_iterations(max_iterations)
  {}

  // Return the approximate lower bound on the distance from x to the set.
  __host__ __device__ __forceinline__
  float operator()( float3 x ) const
  {
  //Warp space around the particle to get the blob-effect.
//    const float part_dist = length( particle - x );
//    const float force = smoothstep( 0.0f, 1.0f, 0.1f / (part_dist*part_dist) ) * 0.2f;
//    const float3 weg = (x - particle) / max(0.01f,part_dist);
//    x -= weg * force;

    // Iterated values.
    float3 zn  = x;//make_float3( x, 0 );
    float4 fp_n = make_float4( 1, 0, 0, 0 );  // start derivative at real 1 (see [2]).

    const float sq_threshold = 2.0f;   // divergence threshold

    float oscillatingTime = sin(global_t / 128.0f );
    float p = (4.0f * abs(oscillatingTime)) + 4.0f; //7.5
    float rad = 0.0f;
    float dist = 0.0f;
    float d = 1.0;

    // Iterate to compute f_n and fp_n for the distance estimator.
    int i = m_max_iterations;
    while( i-- )
    {
//      fp_n = 2.0f * mul( make_float4(zn), fp_n );   // z prime in [2]
//      zn = square( make_float4(zn) ) + c4;         // equation (1) in [1]

      // Stop when we know the point diverges.
      // TODO: removing this condition burns 2 less registers and results in
      //       in a big perf improvement. Can we do something about it?

      rad = length(zn);

      if( rad > sq_threshold )
      {
        dist = 0.5f * rad * logf( rad ) / d;
      }
      else
      {
        float th = atan2( length( make_float3(zn.x, zn.y, 0.0f) ), zn.z );
        float phi = atan2( zn.y, zn.x );
        float rado = pow(rad, p);
        d = pow(rad, p - 1) * (p-1) * d + 1.0;

        float sint = sin(th * p);
        zn.x = rado * sint * cos(phi * p);
        zn.y = rado * sint * sin(phi * p);
        zn.z = rado * cos(th * p);
        zn += x;
      }
    }

    // Distance estimation. Equation (8) from [1], with correction mentioned in [2].
    //const float norm = length( zn );

    //float a = length(x) - 1.0f;
//    float a = dist;
//    float b = sdBox(x, make_float3(1.0f) );

    return dist;

//      float3 p = x;

//    p = pMod(p, 1.0);

//    float d1 = sdBox(p, make_float3(1.0f));

//    float d = sdBox(p, make_float3(1.0f));

//    float s = 1.0f;
//    for(int i = 0; i < 3; i++)
//    {
//        float3 a = myfmod(p * s, 2.0) - 1.0;
//        s *= 3.0;

//        float3 r = 1.0f - 3.0 * myfabs(a);

//        float c = sdCross(r) / s;
//        d = max(d, -c);
//    }

//    float box = sdBox(x, make_float3(4));
//    float d = map(x);

//    return d;

//    float d1 = sdBox(p, make_float3(1.0f) );
//    float d2 = sdBox(p - make_float3(0.6f), make_float3(1.1f) );

//    return max(-d1, d2) / 9.0;

    //return julia_dist;
    //return fminf( julia_dist, part_dist - 0.2f );  // this "renders" the particle as well
  }

  unsigned int m_max_iterations;
};

__device__ bool insideSphere(float3 _point, float3 _center, float _radiusSqr, float* _distance) {
    float distance = length( _point - _center );
    float radius = sqrt(_radiusSqr);
    *_distance = distance;
    if(distance <= radius)
    {
        return true;
    }
    return false;
}

RT_PROGRAM void intersect(int primIdx)
{
  normal = make_float3(0,0,0);

  bool shouldSphereTrace = false;
  float tmin, tmax;
  tmin = 0;
  tmax = RT_DEFAULT_MAX;

  const float sqRadius = 8;

  float distance;
  if( insideSphere(ray.origin, make_float3(0,0,0), sqRadius, &distance) )
  {
//      rtPrintf("Inside sphere : %f\n", distance);
      tmin = 0;
      tmax = RT_DEFAULT_MAX;
      shouldSphereTrace = true;
  }
  else
  {
//      rtPrintf("Outside sphere : %f\n", distance);
      // Push hit to nearest point on sphere
      if( intersectBoundingSphere(ray.origin, ray.direction, sqRadius, tmin, tmax) )
      {
          shouldSphereTrace = true;
      }
  }

  if(shouldSphereTrace)
  {
    JuliaSet distance( max_iterations );
    //distance.m_max_iterations = 64;

    // === Raymarching (Sphere Tracing) Procedure ===

    // XXX inline the sphere tracing procedure here because nvcc isn't
    //     generating the right code i guess

    float3 ray_direction = ray.direction;
    float3 x = (ray.origin ) + tmin * ray_direction;

    float dist_from_origin = tmin;

    // Compute epsilon using equation (16) of [1].
    //float epsilon = max(0.000001f, alpha * powf(dist_from_origin, delta));
    //const float epsilon = 1e-3f;
    const float epsilon = 0.00001;

    //http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
    float fudgeFactor = 0.99;

    // float t = tmin;//0.0;
    // const int maxSteps = 128;
    float dist = 0;

    for( unsigned int i = 0; i < 800; ++i )
    {
      dist = distance( x );

      // Step along the ray and accumulate the distance from the origin.
      x += dist * ray_direction;
      dist_from_origin += dist * fudgeFactor;

      // Check if we're close enough or too far.
      if( dist < epsilon || dist_from_origin > tmax  )
      {
//          rtPrintf("%f, %f, %f\n", ray.origin.x, ray.origin.y, ray.origin.z);
          break;
      }
    }

    // Found intersection?
    if( dist < epsilon )
    {
      if( rtPotentialIntersection( dist_from_origin)  )
      {
        // color HACK
        distance.m_max_iterations = 14;  // more iterations for normal estimate, to fake some more detail
        normal = estimate_normal(distance, x, 0.00001 /*DEL*/);
        geometric_normal = normal;
        shading_normal = normal;
        rtReportIntersection( 0 );
      }
    }
  }
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  const float sz = 1.4f;
  aabb->m_min = make_float3(-sz);
  aabb->m_max = make_float3(sz);
}


//
// Julia set shader.
//

rtBuffer<ParallelogramLight>     lights;

RT_PROGRAM void julia_ah_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
//  prd_shadow.attenuation = make_float3(0);

  rtTerminateRay();
}

rtDeclareVariable(float3, emission_color, , );
RT_PROGRAM void diffuseEmitter(){
    if(current_prd.countEmitted){
        current_prd.result = make_float4(emission_color, 1.0f);
        current_prd.result_nrm = make_float3(0);
    }
    current_prd.done = true;
}

rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
  current_prd_shadow.inShadow = true;
  rtTerminateRay();
}


__device__ float hash(float _seed)
{
    return fract(sin(_seed) * 43758.5453 );
}

//https://www.shadertoy.com/view/MsdGzl
__device__ float3 cosineDirection(float _seed, float3 _n)
{
    // compute basis from normal
    // see http://orbit.dtu.dk/fedora/objects/orbit:113874/datastreams/file_75b66578-222e-4c7d-abdf-f7e255100209/content


    float3 tc = make_float3( 1.0f + _n.z - (_n.x*_n.x),
                             1.0f + _n.z - (_n.y*_n.y),
                             -_n.x * _n.y);
    tc = tc / (1.0f + _n.z);
    float3 uu = make_float3( tc.x, tc.z, -_n.x );
    float3 vv = make_float3( tc.z, tc.y, -_n.y );

    float u = hash( 78.233 + _seed);
    float v = hash( 10.873 + _seed);
    float a = 6.283185 * v;

    return sqrt(u) * (cos(a) * uu + sin(a) * vv) + sqrt(1.0 - u) * _n;
}


//__device__ float myshadow( float3 _origin, float3 _dir )
//{
//    float res = 0.0;

//    float tmax = 12.0;

//    float t = 0.001;
//    for(int i=0; i<80; i++ )
//    {
//        float h = map(ro+rd*t);
//        if( h<0.0001 || t>tmax) break;
//        t += h;
//    }

//    if( t>tmax ) res = 1.0;

//    return res;
//}

rtDeclareVariable(float3,        diffuse_color, , );

typedef rtCallableProgramX<float3()> callT;
rtDeclareVariable(callT, do_work,,);

RT_PROGRAM void diffuse()
{
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

  float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hitpoint = ray.origin + ( t_hit * ray.direction);

  float z1 = rnd(current_prd.seed);
  float z2 = rnd(current_prd.seed);
  float3 p;

  cosine_sample_hemisphere(z1, z2, p);

  float3 v1, v2;
  createONB(ffnormal, v1, v2);

//  current_prd.direction = v1 * p.x + v2 * p.y + ffnormal * p.z;
  current_prd.direction = cosineDirection(current_prd.seed/* + frame_number*/, world_geometric_normal);
  current_prd.attenuation = /*current_prd.attenuation * */diffuse_color; // use the diffuse_color as the diffuse response
  current_prd.countEmitted = false;

  float3 normal_color = (normalize(world_shading_normal)*0.5f + 0.5f)*0.9;

  // @Todo, trace back from the hit to calculate a new sample point?
//  PerRayData_pathtrace backwards_prd;
//  backwards_prd.origin = hitpoint;
//  backwards_prd.direction = -ray.direction;

  // Compute direct light...
  // Or shoot one...
  unsigned int num_lights = lights.size();
  float3 result = make_float3(0.0f);

  for(int i = 0; i < num_lights; ++i)
  {
    ParallelogramLight light = lights[i];

    // Sample random point on geo light
    float z1 = rnd(current_prd.seed);
    float z2 = rnd(current_prd.seed);
    float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;
//    light_pos = make_float3(0, 1000, 0);

//    hitpoint = rtTransformPoint(RT_OBJECT_TO_WORLD, hitpoint);

    float Ldist = length(light_pos - hitpoint);
    float3 L = normalize(light_pos - hitpoint);
    float nDl = dot( shading_normal, L );
    float LnDl = dot( light.normal, L );
    float A = length(cross(light.v1, light.v2));

    // cast shadow ray
    if ( nDl > 0.0f && LnDl > 0.0f )
    {
      PerRayData_pathtrace_shadow shadow_prd;
      shadow_prd.inShadow = false;

//      rtPrintf("(%f, %f, %f) at (%d, %d)\n", L.x, L.y, L.z, launch_index.x, launch_index.y);

      Ray shadow_ray = make_Ray(hitpoint + (shading_normal * 0.01), L, pathtrace_shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_object, shadow_ray, shadow_prd);

      if(!shadow_prd.inShadow)
      {
        float weight= nDl * LnDl * A / (M_PIf*Ldist*Ldist);
        result += light.emission * weight;
      }

//      result += shadow_prd.inShadow ? make_float3(0,0,1) : make_float3(1,0,0);
    }
  }


  current_prd.result = make_float4(result, 1.0);
//  current_prd.result = make_float4( do_work(), 1.0f );
  current_prd.result_nrm = shading_normal;
  current_prd.result_world = hitpoint;
  current_prd.result_depth = t_hit;
  current_prd.done = true;
}

//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------
RT_PROGRAM void miss(){
    current_prd.result = make_float4(0.0, 0.0, 0.0f, 0.0f);
    current_prd.result_nrm = make_float3(0.0, 0.0, 0.0);
    current_prd.result_world = make_float3(0.0, 0.0, 0.0);
    current_prd.result_depth = RT_DEFAULT_MAX;

    current_prd.done = true;
}

//
// Chrome shader for force particle.
//

RT_PROGRAM void chrome_ah_shadow()
{
//  // this material is opaque, so it fully attenuates all shadow rays
  prd_radiance.attenuation = make_float3(0);
//  rtTerminateRay();
}

rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
//  prd_radiance.result = make_float3( tex2D(envmap, u, v) );

  current_prd.done = true;
  current_prd.result = tex2D(envmap, u, v);
  current_prd.result.w = 0.0; // Alpha should be 0 if we missed
  current_prd.result_nrm = make_float3(0.0, 0.0, 0.0);
  current_prd.result_world = make_float3(0.0, 0.0, 0.0);
  current_prd.result_depth = RT_DEFAULT_MAX;
  rtTerminateRay();
}
