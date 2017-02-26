#ifndef __DISTANCEFIELDPRIMITIVES_H__
#define __DISTANCEFIELDPRIMITIVES_H__

#include "romanescomath.h"
#include "GLSL_Functions.h"

/* SDF Primitives */
///
/// \brief sdSphere
/// \param p
/// \param s
/// \return
///
__host__ __device__
inline float sdSphere( float3 p, float s )
{
  return length(p) - s;
}

///
/// \brief sdBox
/// \param p
/// \param _b
/// \return
///
__host__ __device__
inline float sdBox( float3 p, float3 _b )
{
  float3 d = fabs(p) - _b;

  float a = max(d.y, d.z);
  float b = max(d.x, a);

  return min(b, 0.0f) + length( max(d, 0.0f) );
}

///
/// \brief sdCross
/// \param _p
/// \return
///
__device__ float sdfCross(float3 _p)
{
    #define inf 10000.0
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
    return make_float2( cos(a)*v.x + sin(a)*v.y,
                       -sin(a)*v.x + cos(a)*v.y );
}

#endif
