
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

#include "DistanceFieldUtils.h"

using namespace optix;


// References:
// [1] Hart, J. C., Sandin, D. J., and Kauffman, L. H. 1989. Ray tracing deterministic 3D fractals
// [2] http://www.devmaster.net/forums/showthread.php?t=4448


rtDeclareVariable(float3,        eye, , );
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


struct PerRayData_radiance
{
  float3 result;
  float3 result_nrm;
  float3 result_world;
  float result_depth;
  float  importance;
  int iter;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable(rtObject,                         top_object, , );
rtDeclareVariable(float, isect_t, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );


// Quaternion helpers.
static __host__ __device__ float4 mul( float4 a, float4 b )
{
  const float3 a2 = make_float3( a.y, a.z, a.w );
  const float3 b2 = make_float3( b.y, b.z, b.w );
  float3 r;
  r = a.x*b2 + b.x*a2 + cross( a2, b2 );
  return make_float4(
                      a.x*b.x - dot( a2, b2 ),
                      r.x, 
                      r.y, 
                      r.z );
}
static __host__ __device__ float4 square( float4 a )
{
  float3 a2 = make_float3( a.y, a.z, a.w );
  float3 r;
  r = 2*a.x*a2;
  return make_float4(
    a.x*a.x - dot( a2,a2 ),
    r.x, r.y, r.z );
}

// Intersect the bounding sphere of the Julia set.
static __host__ __device__ bool intersectBoundingSphere( float3 o, float3 d, float& tmin, float &tmax )
{
  const float sq_radius = 32.0f;
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

//__host__ __device__
//float udBox( float3 p, float3 b )
//{
//  return length( make_float3(
//                        max( fabs(p.x) - b.x, 0.0),
//                        max( fabs(p.y) - b.y, 0.0),
//                        max( fabs(p.z) - b.z, 0.0)
//                          ) );
//}

//__device__ float maxcomp(float3 _p )
//{
//    return max(_p.x,max(_p.y, _p.z));
//}



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


__device__ float map(float3 _p)
{
    float d = sdBox(_p, make_float3(1.0f));

    float s = 1.0;
    for(int m=0; m<5; m++)
    {
        float3 a = fmod(_p * s, 2.0f) - make_float3(1.0f);
        s *= 3.0;

        float3 r = ( make_float3(1.0) - ( make_float3(3.0) * fabs(a)));

        float c = (float)sdCross(r) / (float)s;

        d = max(d, c);
    }

    return d;
}


__device__ float fracf(float x)
{
    return x - floorf(x);
}

__device__ float3 fracf(float3 p)
{
    return make_float3( p.x - floorf(p.x),
                        p.x - floorf(p.y),
                        p.z - floorf(p.z) );
}

//__device__ float map(float3 _p)
//{
//    float scale = 1.0f;

//    float4 orb = make_float4(1000.0);

//    for(int i=0; i<8; i++)
//    {
//        _p = -1.0 + 2.0 * fracf(0.5f * _p + 0.5f);
////        float3 a = make_float3( fmod(_p.x * s, 2.0f),
////                                fmod(_p.y * s, 2.0f),
////                                fmod(_p.z * s, 2.0f)) - 1.0f;
//        float r2 = dot(_p, _p);

//        orb = make_float4( min( orb.x, abs(_p.x) ),
//                           min( orb.y, abs(_p.y) ),
//                           min( orb.z, abs(_p.z) ),
//                           min( orb.w, r2 ) );

//        float k = max( r2, 0.1f);
//        _p *= k;
//        scale *= k;
//    }

//    return 0.25 * abs(_p.y) / scale;
//}


struct JuliaSet
{
  __host__ __device__
  JuliaSet(const unsigned int max_iterations) : m_max_iterations(max_iterations)
  {}

  // Return the approximate lower bound on the distance from x to the set.
  __host__ __device__ __forceinline__
  float operator()( float3 x ) const
  {
    // Warp space around the particle to get the blob-effect.
//    const float part_dist = length( particle - x );
//    const float force = smoothstep( 0.0f, 1.0f, 0.1f / (part_dist*part_dist) ) * 0.2f;
//    const float3 weg = (x - particle) / max(0.01f,part_dist);
//    x -= weg * force;

//    // Iterated values.
//    float3 zn  = x;//make_float3( x, 0 );
//    float4 fp_n = make_float4( 1, 0, 0, 0 );  // start derivative at real 1 (see [2]).

//    const float sq_threshold = 2.0f;   // divergence threshold

//    float oscillatingTime = sin(global_t / 256.0f );
//    float p = (5.0f * abs(oscillatingTime)) + 3.0f; //8;
//    float rad = 0.0f;
//    float dist = 0.0f;
//    float d = 1.0;

//    // Iterate to compute f_n and fp_n for the distance estimator.
//    int i = m_max_iterations;
//    while( i-- )
//    {
////      fp_n = 2.0f * mul( make_float4(zn), fp_n );   // z prime in [2]
////      zn = square( make_float4(zn) ) + c4;         // equation (1) in [1]

//      // Stop when we know the point diverges.
//      // TODO: removing this condition burns 2 less registers and results in
//      //       in a big perf improvement. Can we do something about it?

//      rad = length(zn);

//      if( rad > sq_threshold )
//      {
//        dist = 0.5f * rad * logf( rad ) / d;
//      }
//      else
//      {
//        float th = atan2( length( make_float3(zn.x, zn.y, 0.0f) ), zn.z );
//        float phi = atan2( zn.y, zn.x );
//        float rado = pow(rad, p);
//        d = pow(rad, p - 1) * (p-1) * d + 1.0;

//        float sint = sin(th * p);
//        zn.x = rado * sint * cos(phi * p);
//        zn.y = rado * sint * sin(phi * p);
//        zn.z = rado * cos(th * p);
//        zn += x;
//      }
//    }

//    // Distance estimation. Equation (8) from [1], with correction mentioned in [2].
//    //const float norm = length( zn );

//    //float a = length(x) - 1.0f;
//    float a = dist;
//    float b = sdBox(x, make_float3(1.0f) );

    //return dist;

      float3 p = x;

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

    float d = map(p);

    return d;

//    float d1 = sdBox(p, make_float3(1.0f) );
//    float d2 = sdBox(p - make_float3(0.6f), make_float3(1.1f) );

//    return max(-d1, d2) / 9.0;

    //return julia_dist;
    //return fminf( julia_dist, part_dist - 0.2f );  // this "renders" the particle as well
  }

  unsigned int m_max_iterations;
};


RT_PROGRAM void intersect(int primIdx)
{
  normal = make_float3(0,0,0);

  float tmin, tmax;
  if( intersectBoundingSphere(ray.origin, ray.direction, tmin, tmax) )
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


    // float t = tmin;//0.0;
    // const int maxSteps = 128;
    float dist = 0;

    for( unsigned int i = 0; i < 800; ++i )
    {
      dist = distance( x );

      // Step along the ray and accumulate the distance from the origin.
      x += dist * ray_direction;
      dist_from_origin += dist;

      // Check if we're close enough or too far.
      if( dist < epsilon || dist_from_origin > tmax  )
      {
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
        normal = estimate_normal(distance, x, DEL);
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

RT_PROGRAM void julia_ch_radiance()
{
  const float3 p = ray.origin + isect_t * ray.direction;

  // ambient occlusion
  JuliaSet distance( max_iterations );
  float occlusion = 1.f;
  float fact = 7.f;
  const float delta = 0.05f;
  for( int i=0; i<4; ++i ) {
    const float dist = delta * i;
    occlusion -= fact * (dist - distance(p+dist*normal));
    fact *= 0.5f;
  }
  occlusion += 0.3f;
  occlusion *= occlusion;
  occlusion = clamp( occlusion, 0.2f, 1.0f );

//   base colors
  float3 red   = normal*0.5f + make_float3(0.5f);
  float3 green = red;
  float3 blue  = red;


//   red/orange
  red.x = abs(normal.x)*0.5f + 0.5f;
  red.x = max( red.x, 0.1f );
  red = red * make_float3( 0.8f, 0.1f+red.x, 0.1f );
  red.y += 0.2f * red.x;
  red.x += 0.6f;
  red.x *= max(occlusion,0.8f);
  red.y *= occlusion;
  red.z *= occlusion;
  
  // green
  green.x = abs(normal.x)*0.5f + 0.5f;
  green.z = -abs(normal.z)*0.5f + 0.5f;
  green.y = green.y * 0.7f + 0.3f;
  green = green * make_float3( 0.9f*green.y*green.y, 1.0f, 0.2f );
  green.x += 0.2f;
  green.x *= green.x;
  green.x *= occlusion;
  green.y = max(0.3f,green.y*occlusion);
  green.z *= occlusion;

  // blue
  blue.x = abs(normal.x)*0.5f + 0.5f;
  blue.y = abs(normal.y)*0.5f + 0.5f;
  blue.z = -abs(normal.z)*0.5f + 0.5f;
  blue.z = blue.z * 0.7f + 0.3f;
  blue.x += 0.2f;
  blue.y += 0.2f;
  blue = blue * make_float3( 0.9f*blue.y*blue.y, 1.0f*blue.z*blue.y, 1.0f );
  blue.z += 0.3f;
  blue.x *= blue.z * max(0.3f,occlusion);
  blue.y *= occlusion;
  blue.z = blue.z * max(0.6f,occlusion);

  // select color
  float3 c0 = green;
  float3 c1 = red;
  float ct = color_t;
  if( color_t > 1.0f ) {
    c0 = red;
    c1 = blue;
    ct -= 1.0f;
  }
  float3 result = green; //dot(p,p) > ct*3.0f ? c0 : c1;

  // add glow close to particle
  const float part_dist = length( p-particle );
  const float glow = 1.0f - smoothstep( 0.0f, 1.0f, part_dist );
  result = result + make_float3(glow*0.4f);

  // add phong highlight
  const float3 l = make_float3( 1, 3, 1 );
  const float3 h = normalize( l - ray.direction );
  const float ndh = dot( normal, h );
  if( ndh > 0.0f ) {
    result = result + make_float3( 0.6f * occlusion * pow(ndh,20.0f) );
  }

  //  float magic_ambient_occlusion(const float3 &x, const float3 &n,
//                                const float del,
//                                const float k,
//                                Distance distance)

//  float ao = magic_ambient_occlusion( );

  //result = make_float3( occlusion );

  // Reflection (disabled, doesn't look too great)

  //if( prd_radiance.depth < 5 )
  //if( prd_radiance.depth < 5 )
  {
      PerRayData_radiance new_prd;
      new_prd.importance = prd_radiance.importance;
      new_prd.depth = prd_radiance.depth + 1;
      new_prd.result = make_float3(1,0,0);

      float3 refl = make_float3(0,0,0);
      refl = reflect( ray.direction, normal );
      const optix::Ray refl_ray = optix::make_Ray( p, refl, 0, 1e-3f, RT_DEFAULT_MAX );
      rtTrace( top_object, refl_ray, new_prd );

      PerRayData_radiance new_prd2;
      new_prd2.importance = prd_radiance.importance;
      new_prd2.depth = prd_radiance.depth + 1;
      new_prd2.result = make_float3(1,0,0);

      float3 refr = make_float3(0,0,0);
      refract( refr, ray.direction, normal, 1.3);
      const optix::Ray refr_ray = optix::make_Ray( p, refr, 0, 1e-3f, RT_DEFAULT_MAX );
      rtTrace( top_object, refr_ray, new_prd2 );

//      result = (result * occlusion) + new_prd.result;
      result = lerp(new_prd.result + new_prd2.result, result, 0.9);//lerp( new_prd.result * occlusion, result, 0 );
  }

  prd_radiance.result = result;
  prd_radiance.result_nrm = normal;//normalize( rtTransformNormal(RT_OBJECT_TO_WORLD, normal) )*0.5f + 0.5f;
  prd_radiance.result_world = p;
  prd_radiance.result_depth = length(p - eye) / 100.0;
}

RT_PROGRAM void julia_ah_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  
  rtTerminateRay();
}


//
// Chrome shader for force particle.
//

RT_PROGRAM void chrome_ch_radiance()
{
  float3 dir = ray.direction;
  const float3 hit_point = ray.origin + isect_t * dir;

  if( prd_radiance.depth < 3 )
  {
    PerRayData_radiance new_prd;             
    new_prd.importance = prd_radiance.importance;
    new_prd.depth = prd_radiance.depth + 1;
    
    const float3 refl = reflect( dir, shading_normal );
    const optix::Ray refl_ray = optix::make_Ray( hit_point, refl, 0, 1e-3f, RT_DEFAULT_MAX );
    rtTrace( top_object, refl_ray, new_prd );
    const float fresnel = fresnel_schlick( dot(shading_normal,-dir), 5.0f, 0.3f, 1.0f );
    const float diff = (shading_normal.y+1.f) * 0.5f;
    prd_radiance.result = new_prd.result * fresnel
      + make_float3(diff*diff*diff*0.1f);
  } else {
    prd_radiance.result = make_float3( 0 );
  }
}

RT_PROGRAM void chrome_ah_shadow()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  rtTerminateRay();
}

rtTextureSampler<float4, 2> envmap;
RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PIf * 0.5f -  acosf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     = 0.5f * ( 1.0f + sin(phi) );
  prd_radiance.result = make_float3( tex2D(envmap, u, v) );
}
