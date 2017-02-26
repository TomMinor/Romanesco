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


class IFSTest : public DistanceEstimator
{
public:
    __device__ inline IFSTest(const unsigned int _maxIterations,
                       float _scale = 2.0f,
                       float3 _offset = make_float3(0.92858,0.92858,0.32858),
                       float _fudgeFactor = 0.8f,
                       float3 _limits = make_float3(2.0f)
                       )
        : DistanceEstimator(_maxIterations)
    {
        m_scale = _scale;
        m_offset = _offset;
        m_fudgeFactor = _fudgeFactor;
        m_limits = _limits;
    }

    __device__ inline virtual void evalParameters()
    {
        // m_morphOffset = ?
    }

    __device__ inline virtual float evalDistance(float3 _p)
    {
        float a = map(_p) * m_fudgeFactor;
        _p.y += 1;
        float b = sdBox(_p, m_limits);
        return max(a,b);
    }

private:
    __device__ inline float map(float3 _p)
    {
        float t = m_time / 18.0f;
        t = tan(t);
        Mandelbulb sdf(m_maxIterations, t);

        float global_t = m_time;
        float3 offset = make_float3(1.0 + 0.2f * cos( 1.0f * (global_t / 5.7f)),
                                    1.0,
                                    0.3 + 0.1f * (cos( 1.0f * (global_t / 1.7f)))
                                    );

        float3 z = _p;
//        float d2 = sdf.evalDistance(z);
//        z.x -= global_t * 0.01f;
        z = translateHook(0, z);
        z = rotateHook(0, z);
        z = scaleHook(0, z);

//        z.x = fmod(z.x, 3.5f);

        z = fabs( make_float3(1.0f) - fmod(z, 2.0f));
//        z.x = fabs(z.x + m_offset.x) - m_offset.x;
//        z.x = fabs(z.x + offset.x) - offset.x;


        float d = 1000.0f;
        for(int n = 0; n < m_maxIterations; ++n)
        {
            ///@todo rotate
            ///
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

            z = z * m_scale - offset * (m_scale - 1.0);

            float2 tmp = make_float2(z.y, z.z);
    //        Matrix4x4 rotation = Matrix4x4::rotate( radians(-global_t / 18.0f), make_float3(1, 0, 0) );
    //        float3 r = applyRotation( make_float3(z.y, z.z, 0.0f),  rotation);

            float2 r = rotate(tmp, -global_t / 18.0f);
            z.y = r.x;
            z.z = r.y;

            d = min(d, length(z) * powf(m_scale, -float(n+1)));
        }

//        d = max(d, d2);
        return d;
    }

protected:
    float m_scale;
    float m_fudgeFactor;
    float3 m_offset;
    float3 m_morphOffset;
    float3 m_limits;
};



#endif
