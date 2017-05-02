#pragma once

#include "DistanceFieldAdvancedPrimitives.h"

class Mandelbulb : public DistanceEstimator
{
public:
    __device__ Mandelbulb(const unsigned int _maxIterations, float _power = 8.0f) : DistanceEstimator(_maxIterations)
    {
        m_power = _power;
    }

    __device__ void setPower(float _power)
    {
        m_power = _power;
    }

    __device__ inline virtual void evalParameters()
    {
        float oscillatingTime = sin(m_time / 40.0f );
        m_power = (1.0f * oscillatingTime) + 7.0f;
    }

    __device__ inline virtual float evalDistance(float3 _p)
    {
        float3 zn  = _p;
        const float sq_threshold = 2.0f;   // divergence threshold

        float p = m_power;
        float rad = 0.0f;
        float dist = 0.0f;
        float d = 1.0;

        //            z = z * m_scale - offset * (m_scale - 1.0);

        //            float2 tmp = make_float2(z.y, z.z);

        zn = translateHook(0, zn);
        zn = rotateHook(0, zn);
        zn = scaleHook(0, zn);


        float m_scale = 1.0f;
        float3 offset = make_float3(0.92858,0.92858,0.32858);

        const float s = 0.9f;
        float k = 1.0f;
        float m0 = 1e10, m1 = 1e10, m2 = 1e10;

        SphereTrap trapA;

        // Iterate to compute f_n and fp_n for the distance estimator.
        int i = m_maxIterations;
        while( i-- )
        {
            trapA.trap(zn);

            m0 = min(m0, dot(zn, zn) / (k * k) );
            m1 = min(m1, trapA.getTrapValue() );
            m2 = length( make_float3( zn.z, zn.x, 0.0f) - make_float3(0.25, 0.25, 0.0)) - 0.3; // <- tube forms

            rad = length(zn);

//          zn = zn * m_scale - offset * (m_scale - 1.0);

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
                zn += _p;
            }

            k *= s;

//          float2 r = rotate(tmp, -global_t / 18.0f);
//          Matrix4x4 rotation = Matrix4x4::rotate( radians(-m_time / 18.0f), make_float3(1, 0, 0) );
//          float3 r = applyTransform( make_float3(zn.y, zn.z, 0.0f),  rotation);
//          zn.y = r.x;
//          zn.z = r.y;
        }

        setTrap0( m0 );
        setTrap1( m1 );
        setTrap2( m2 );

        return dist;
    }

private:
    float m_power;

};
