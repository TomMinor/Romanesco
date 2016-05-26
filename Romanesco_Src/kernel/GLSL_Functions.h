#ifndef __GLSL_FUNCTIONS
#define __GLSL_FUNCTIONS

///
/// \brief This header provides functions for common operations and overloads them for float3's, etc. in a way similiar to the GLSL spec.
/// \author Tom Minor
///

#ifdef ROMANESCO_RUNTIME_COMPILE
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil_math.h>
#else
#include <optix_math.h>
#endif


__host__ __device__ float degrees(float _radians)
{
    return (_radians * (180.0f / M_PI));
}

__host__ __device__ float radians(float _degrees)
{
    return (_degrees * (M_PI / 180.0f));
}

//------------------------------------- fmod ------------------------------------------------

 __host__ __device__
inline float4 fmod(float4 _p, float _s)
{
    return make_float4( fabs( fmod(_p.x, _s)),
                        fabs( fmod(_p.y, _s)),
                        fabs( fmod(_p.z, _s)),
                        fabs( fmod(_p.w, _s))  );
}

///
/// \brief fmod Returns the floating point remainder of a float3
/// \param _p
/// \param _s
/// \return
///
 __host__ __device__
inline float3 fmod(float3 _p, float _s)
{
    // http://stackoverflow.com/questions/7610631/glsl-mod-vs-hlsl-fmod
    return make_float3( fabs( fmod(_p.x, _s)),
                        fabs( fmod(_p.y, _s)),
                        fabs( fmod(_p.z, _s))  );
}

 __host__ __device__
inline float2 fmod(float2 _p, float _s)
{
    return make_float2( fabs( fmod(_p.x, _s)),
                        fabs( fmod(_p.y, _s)) );
}

//------------------------------------- \fmod ------------------------------------------------


//------------------------------------- frac ------------------------------------------------

 static __device__ inline float3 powf(float3 a, float exp)
 {
   return make_float3(powf(a.x, exp), powf(a.y, exp), powf(a.z, exp));
 }

///
/// \brief frac Returns the fractional component of a float
/// \param x
/// \return
///
__host__ __device__
inline float fract(float x)
{
    return x - floorf(x);
}

///
/// \brief frac Returns the fractional component of a float2
/// \param p
/// \return
///
__host__ __device__
inline float2 frac(float2 p)
{
    return make_float2( p.x - floorf(p.x),
                        p.x - floorf(p.y));
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

//------------------------------------- smin ------------------------------------------------

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

//------------------------------------- \smin ------------------------------------------------

//------------------------------------- max ------------------------------------------------

///
/// \brief max
/// \param _a
/// \param _b
/// \return
///
__host__ __device__
inline float4 max(float4 _a, float _b)
{
  float a = fmaxf(_a.x, _b);
  float b = fmaxf(_a.y, _b);
  float c = fmaxf(_a.z, _b);
  float d = fmaxf(_a.w, _b);

  return make_float4(a,b,c,d);
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
/// \brief max
/// \param _a
/// \param _b
/// \return
///
__host__ __device__
inline float2 max(float2 _a, float _b)
{
  float a = fmaxf(_a.x, _b);
  float b = fmaxf(_a.y, _b);

  return make_float2(a,b);
}

//------------------------------------- \max ------------------------------------------------

//------------------------------------- min ------------------------------------------------


///
/// \brief min
/// \param _a
/// \param _b
/// \return
__host__ __device__
inline float4 min(float4 _a, float _b)
{
  float a = fminf(_a.x, _b);
  float b = fminf(_a.y, _b);
  float c = fminf(_a.z, _b);
  float d = fminf(_a.w, _b);

  return make_float4(a,b,c,d);
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
/// \brief min
/// \param _a
/// \param _b
/// \return
__host__ __device__
inline float2 min(float2 _a, float _b)
{
  float a = fminf(_a.x, _b);
  float b = fminf(_a.y, _b);

  return make_float2(a,b);
}

//------------------------------------- \min ------------------------------------------------


//------------------------------------- Swizzle ------------------------------------------------



//------------------------------------- \Swizzle ------------------------------------------------


// Some functions are defined in cutil_math but apparently don't include properly when compiling this with optix included
#ifndef ROMANESCO_RUNTIME_COMPILE

///
/// \brief fabs float3 overload of fabs
/// \param _p
/// \return
///
__device__
inline float3 fabs(float3 _p)
{
    return make_float3( fabs(_p.x), fabs(_p.y), fabs(_p.z) );
}

#endif






#endif
