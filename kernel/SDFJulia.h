#pragma once

#include "DistanceFieldAdvancedPrimitives.h"

// Quaternion helpers.
static __host__ __device__ float4 mul( float4 a, float4 b )
{
  const float3 a2 = make_float3( a.y, a.z, a.w );
  const float3 b2 = make_float3( b.y, b.z, b.w );
  float3 r;
  r = a.x*b2 + b.x*a2 + cross( a2, b2 );
  return make_float4(
    a.x*b.x - dot( a2, b2 ),
    r.x, r.y, r.z );
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

class Julia : public DistanceEstimator
{
public:
    __device__ Julia(const unsigned int _maxIterations) : DistanceEstimator(_maxIterations)
    {
        setC(make_float4(-0.5f, 0.1f, 0.2f, 0.3f));
    }

    __device__ inline void setC(float4 _c)
    {
        m_c = _c;
    }

    __device__ inline virtual void evalParameters()
    {

    }

    __device__ inline virtual float evalDistance(float3 _p)
    {
        // Iterated values.
        float4 d = make_float4( 1, 0, 0, 0);  // start derivative at real 1 (see [2]).

        const float sq_threshold = 16.0f;   // divergence threshold

//        _p = translateHook(0, _p);
//        _p = rotateHook(0, _p);
//        _p = scaleHook(0, _p);

        float4 zn = make_float4(_p, 0.0);

        float4 z  = make_float4(_p, 0.0f);
        float md2 = 1.0f;
        float mz2 = dot(z,z);


        float m_scale = 1.0f;
        float3 offset = make_float3(0.92858,0.92858,0.32858);

        const float s = 0.9f;
        float k = 1.0f;
        float m0 = 1e10, m1 = 1e10, m2 = 1e10;

        SphereTrap trapA;

//        const float4 c = make_float4(0.5, 3.9, 1.4, 1.1);

        // Iterate to compute f_n and fp_n for the distance estimator.
        int i = m_maxIterations;
        while( i-- )
        {
            trapA.trap( make_float3(zn.x, zn.y, zn.z) );

            m0 = min(m0, dot(zn, zn) / (k * k) );
            m1 = min(m1, trapA.getTrapValue() );
            m2 = length( make_float3( zn.z, zn.x, 0.0f) - make_float3(0.25, 0.25, 0.0)) - 0.3; // <- tube forms

            // |dz|^2 -> 4*|dz|^2
            md2 *= 4.0f * mz2;

            // z -> z2 + c
            float3 tmp = 2.0*z.x*make_float3(z.y, z.z, z.w);
            z = make_float4( z.x*z.x - dot(make_float3(z.y, z.z, z.w),
                                           make_float3(z.y, z.z, z.w)),
                             tmp.x, tmp.y, tmp.z) + m_c;

            mz2 = dot(z,z);
            if(mz2 > 4.0f)
            {
                break;
            }

//            if( dot(zn,zn) > sq_threshold )
//            {
//                break;
//            }
//            else
//            {
//                d = 2.0f * mul( zn, d);
//                zn = square( d ) + c4;
//            }

            k *= s;
        }

        setTrap0( m0 );
        setTrap1( m1 );
        setTrap2( m2 );

//        float rad = length(zn);
//        float dist = 0.5f * rad * logf(rad) / length(d);
        return 0.25f * sqrt(mz2/md2) * log(mz2);
    }

private:
    float4 m_c;
};
