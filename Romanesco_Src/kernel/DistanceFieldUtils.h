#ifndef __DISTANCE_FIELD_UTILS
#define __DISTANCE_FIELD_UTILS

#ifdef ROMANESCO_RUNTIME_COMPILE
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil_math.h>
#else
#include <optix_math.h>
#endif

//http://stackoverflow.com/questions/7610631/glsl-mod-vs-hlsl-fmod

///
/// \brief fmod Returns the floating point remainder of a float3
/// \param _p
/// \param _s
/// \return
///
 __host__ __device__
inline float3 fmod(float3 _p, float _s)
{
    return make_float3( fabs( fmod(_p.x, _s)),
                        fabs( fmod(_p.y, _s)),
                        fabs( fmod(_p.z, _s))  );
}

///
/// \brief frac Returns the fractional component of a float
/// \param x
/// \return
///
__host__ __device__
inline float frac(float x)
{
    return x - floorf(x);
}

///
/// \brief frac Returns the fractional component of a float3
/// \param p
/// \return
///
__host__ __device__
inline float3 frac(float3 p)
{
    return make_float3( p.x - floorf(p.x),
                        p.x - floorf(p.y),
                        p.z - floorf(p.z) );
}

///
/// \brief smin Smooth minimum blend between two volumes
/// \param a
/// \param b
/// \param k
/// \return
///
__host__ __device__
inline float smin( float a, float b, float k )
{
    float h = clamp( 0.5f + 0.5f *  (b - a) / k, 0.0f, 1.0f );
    return lerp( b, a, h ) - k*h*(1.0-h);
}

///
/// \brief max
/// \param _a
/// \param _b
/// \return
///
__host__ __device__
inline float3 max(float3 _a, float _b)
{
  float a = fmaxf(_a.x, _b);
  float b = fmaxf(_a.y, _b);
  float c = fmaxf(_a.z, _b);

  return make_float3(a,b,c);
}

///
/// \brief min
/// \param _a
/// \param _b
/// \return
__host__ __device__
inline float3 min(float3 _a, float _b)
{
  float a = fminf(_a.x, _b);
  float b = fminf(_a.y, _b);
  float c = fminf(_a.z, _b);

  return make_float3(a,b,c);
}

///
/// \brief fabs float3 overload of fabs
/// \param _p
/// \return
///
//__device__
//inline float3 fabs(float3 _p)
//{
//    return make_float3( fabs(_p.x), fabs(_p.y), fabs(_p.z) );
//}


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



#endif
