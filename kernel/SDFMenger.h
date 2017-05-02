#pragma once

#include "DistanceFieldAdvancedPrimitives.h"

class MengerSponge : public DistanceEstimator
{
public:
    __device__ MengerSponge(const unsigned int _maxIterations,
                            unsigned int _depth = 3)
        : DistanceEstimator(_maxIterations)
    {
        m_depth = _depth;
    }

    __device__ inline virtual void evalParameters()
    {
        m_rotate = make_float3(m_time);
    }

    __device__ inline virtual float evalDistance(float3 _p)
    {
        float d = sdBox(_p, make_float3(1.0f));

        float s = 1.0;
        for(int m=0; m<m_depth; m++)
        {
            Matrix4x4 rotX = Matrix4x4::rotate( radians( m_rotate.x ) , make_float3(1,0,0) );
            Matrix4x4 rotY = Matrix4x4::rotate( radians( m_rotate.y ) , make_float3(0,1,0) );
            Matrix4x4 rotZ = Matrix4x4::rotate( radians( m_rotate.z ) , make_float3(0,0,1) );

            _p = applyTransform(_p, rotX);
            _p = applyTransform(_p, rotY);
            _p = applyTransform(_p, rotZ);

            float3 a = fmod(_p * s, 2.0f) - make_float3(1.0f);
            s *= 3.0;

            float3 r = ( make_float3(1.0) - ( make_float3(3.0) * fabs(a)));

            float c = (float)sdfCross(r) / (float)s;

            d = max(d, c);
        }

        return d;
    }

protected:
    float3 m_rotate;
    unsigned int m_depth;
};
