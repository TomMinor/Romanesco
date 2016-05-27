#ifndef __DISTANCEFIELD_H__
#define __DISTANCEFIELD_H__

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;

// Helper macro to convert to vec4 for the purpose of a rotation
#define applyTransform(p,rot) make_float3(make_float4(p, 1.0f) * rot);

inline __device__ float lengthSqr(float3 _v)
{
    return dot(_v, _v);
}

static __device__ bool intersectBoundingSphere( float3 o, float3 d, float sqRadius, float& tmin, float &tmax )
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

template<typename SDF>
static __device__ float3 finiteDifference(SDF sdf, const float3 &x, const float eps = 1e-6f)
{
  float dx = sdf.evalDistance(x + make_float3(eps,    0,   0)) - sdf.evalDistance(x - make_float3(eps,   0,   0));
  float dy = sdf.evalDistance(x + make_float3(  0,  eps,   0)) - sdf.evalDistance(x - make_float3(  0, eps,   0));
  float dz = sdf.evalDistance(x + make_float3(  0,    0, eps)) - sdf.evalDistance(x - make_float3(  0,   0, eps));

  return make_float3(dx, dy, dz);
}

template<typename SDF>
static __device__ float3 calculateNormal(const SDF distance,
                       const float3 &x,
                       const float epsilon = 1e-3f)
{
  return normalize(finiteDifference(distance, x, epsilon));
}

#endif
