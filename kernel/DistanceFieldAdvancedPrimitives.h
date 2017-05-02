#ifndef DISTANCEFIELDADVANCEDPRIMITIVES_H__
#define DISTANCEFIELDADVANCEDPRIMITIVES_H__

#include "romanescomath.h"
#include "DistanceFieldMaths.h"
#include "DistanceFieldPrimitives.h"
#include "DistanceFieldTraps.h"

#define TOTALXFORMHOOKS 3
#define TOTALTRAPS 3

///
/// \brief The DistanceEstimator class is an interface for the more complex fractal surfaces that require state and many parameters
///
class DistanceEstimator
{
public:
    __device__ DistanceEstimator(const unsigned int _maxIterations)
    {
        m_maxIterations = _maxIterations;
        m_time = 0.0f;

        m_trap0 = 0.0f;
        m_trap1 = 0.0f;
        m_trap2 = 0.0f;

        // Initialise default hook values
        for(uint i = 0; i < TOTALXFORMHOOKS; i++)
        {
            setScaleHook(i, make_float3(1.0f));
            setRotateHook(i, make_float3(0.0f));
            setTranslateHook(i, make_float3(0.0f));
        }
    }

    __device__ inline virtual void evalParameters()  = 0;

    __device__ inline virtual float evalDistance(float3 _p) = 0;

    __device__ inline void setTrap0(float _t)      {   m_trap0 = _t; }
    __device__ inline void setTrap1(float _t)      {   m_trap1 = _t; }
    __device__ inline void setTrap2(float _t)      {   m_trap2 = _t; }

    __device__ inline float getTrap0()      {   return m_trap0; }
    __device__ inline float getTrap1()      {   return m_trap1; }
    __device__ inline float getTrap2()      {   return m_trap2; }

    __device__ inline unsigned int getMaxIterations()
    {
        return m_maxIterations;
    }

    __device__ inline void setMaxIterations(unsigned int _iterations)
    {
        m_maxIterations = _iterations;
    }

    __device__ inline void setTime(float _t)
    {
        m_time = _t;
    }

    __device__ inline float3 scaleHook(uint _idx, float3 _v)
    {
        if(_idx > (TOTALXFORMHOOKS - 1)) {
            return _v;
        }

        float3 amount = m_scale[_idx];

        // Special case for uniform scale
        if( amount == make_float3(1.0f) )
        {
            return _v;
        }

        Matrix4x4 transform = Matrix4x4::scale(amount);
        _v = applyTransform(_v, transform);

        return _v;
    }

    __device__ inline float3 rotateHook(uint _idx, float3 _v)
    {
        if(_idx > (TOTALXFORMHOOKS - 1)) {
            return _v;
        }

        float3 amount = m_rotate[_idx];

        // Special case for no rotation
        if( amount == make_float3(0.0f) )
        {
            return _v;
        }

        Matrix4x4 transformX = Matrix4x4::rotate(amount.x, make_float3(1,0,0));
        Matrix4x4 transformY = Matrix4x4::rotate(amount.y, make_float3(0,1,0));
        Matrix4x4 transformZ = Matrix4x4::rotate(amount.z, make_float3(0,0,1));
        _v = applyTransform(_v, transformX * transformY * transformZ);

        return _v;
    }

    __device__ inline float3 translateHook(uint _idx, float3 _v)
    {
//        if(_idx > (TOTALXFORMHOOKS - 1)) {
//            return _v;
//        }

        float3 amount = m_translate[_idx];

        // Special case for no translation
//        if( amount == make_float3(0.0f) )
//        {
//            return _v;
//        }

//        Matrix4x4 transform = Matrix4x4::translate(amount);
        _v += amount;//applyTransform(_v, transform);

        return _v;
    }

    __device__ inline void setScaleHook(uint _idx, float3 _v)
    {
        m_scale[_idx] = _v;
    }

    __device__ inline void setRotateHook(uint _idx, float3 _v)
    {
        m_rotate[_idx] = _v;
    }

    __device__ inline void setTranslateHook(uint _idx, float3 _v)
    {
        m_translate[_idx] = _v;
    }

protected:
    unsigned int m_maxIterations;
    float m_time;
    float m_trap0, m_trap1, m_trap2;

    float3 m_scale[TOTALXFORMHOOKS];
    float3 m_rotate[TOTALXFORMHOOKS];
    float3 m_translate[TOTALXFORMHOOKS];
};

#endif
