#ifndef DISTANCEFIELDADVANCEDPRIMITIVES_H__
#define DISTANCEFIELDADVANCEDPRIMITIVES_H__


#include "DistanceFieldMaths.h"
#include "DistanceFieldPrimitives.h"

using namespace optix;

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
    }

    __device__ virtual void evalParameters()  = 0;

    __device__ virtual float evalDistance(float3 _p) = 0;

    __device__ unsigned int getMaxIterations()
    {
        return m_maxIterations;
    }

    __device__ void setMaxIterations(unsigned int _iterations)
    {
        m_maxIterations = _iterations;
    }

    __device__ void setTime(float _t)
    {
        m_time = _t;
    }

protected:
    unsigned int m_maxIterations;
    float m_time;
};



class Mandelbulb : public DistanceEstimator
{
public:
    __device__ Mandelbulb(const unsigned int _maxIterations, float _power = 8.0f) : DistanceEstimator(_maxIterations)
    {
        m_power = _power;
    }

    __device__ virtual void evalParameters()
    {
        // Update power, etc from the UI?
    }

    __device__ virtual float evalDistance(float3 _p)
    {
        float3 zn  = _p;
        const float sq_threshold = 2.0f;   // divergence threshold

        float oscillatingTime = sin(m_time / 40.0f );
        float p = (2.0f * oscillatingTime) + 6.0f; //7.5
        float rad = 0.0f;
        float dist = 0.0f;
        float d = 1.0;

        // Iterate to compute f_n and fp_n for the distance estimator.
        int i = m_maxIterations;
        while( i-- )
        {
          rad = length(zn);

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
        }

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

    __device__ virtual void evalParameters()
    {
        m_rotate = make_float3(m_time);
    }

    __device__ virtual float evalDistance(float3 _p)
    {
        float d = sdBox(_p, make_float3(1.0f));

        float s = 1.0;
        for(int m=0; m<m_depth; m++)
        {
            Matrix4x4 rotX = Matrix4x4::rotate( radians( m_rotate.x ) , make_float3(1,0,0) );
            Matrix4x4 rotY = Matrix4x4::rotate( radians( m_rotate.y ) , make_float3(0,1,0) );
            Matrix4x4 rotZ = Matrix4x4::rotate( radians( m_rotate.z ) , make_float3(0,0,1) );

            _p = applyRotation(_p, rotX);
            _p = applyRotation(_p, rotY);
            _p = applyRotation(_p, rotZ);

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
    __device__ IFSTest(const unsigned int _maxIterations,
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

    __device__ virtual void evalParameters()
    {
        // m_morphOffset = ?
    }

    __device__ virtual float evalDistance(float3 _p)
    {
        float a = this->map(_p) * m_fudgeFactor;
        _p.y += 1;
        float b = sdBox(_p, m_limits);
        return max(a,b);
    }

private:
    __device__ float map(float3 _p)
    {
        float global_t = 0.0f;
        float3 offset = make_float3(1.0 + 0.2f * cos( 1.0f * (global_t / 5.7f)),
                                    1.0,
                                    0.3 + 0.1f * (cos( 1.0f * (global_t / 1.7f)))
                                    );

        float3 z = _p;
        z.x -= global_t * 0.01f;
        z.x = fmod(z.x, 1.0f);

        z = fabs( 1.0 - fmod(z, 2.0));
        z.x = fabs(z.x + m_offset.x) - m_offset.x;


        float d = 1000.0f;
        for(int n = 0; n < m_maxIterations; ++n)
        {
            ///@todo rotate

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
