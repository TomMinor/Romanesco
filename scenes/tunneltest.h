#include "romanescocore.h"

//#define MANDELTRAP

class TunnelTest : public DistanceEstimator
{
public:
    __device__ inline TunnelTest(const unsigned int _maxIterations,
                       float _scale = 2.0f,
                       float3 _offset = make_float3(0.92858,0.92858,0.32858),
                       float _fudgeFactor = 0.8f,
                       float3 _limits = make_float3(7.0f, 2.0f, 1.0f)
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
        float length = 3.5f;
        float3 z = _p;
        float3 x = _p;

        float a = map( translateHook(0, z) ) * m_fudgeFactor;
        x.y += 1;
        float b = sdBox(x  + make_float3(length, 0.0f, 0.0f), m_limits);
        return max(a,b);
    }

private:
    /// http://stackoverflow.com/questions/3451553/value-remapping
    __device__ float remap(float value, float low1, float high1, float low2, float high2)
    {
        return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
    }

    //// Geometric orbit trap
     __device__ float trap(float3 p)
     {
         Mandelbulb sdf( 8 );
         sdf.setTime(m_time);
        sdf.evalParameters();
         sdf.setPower( 4/*(3.0f * abs(sin(m_time / 40.f))) + 5.0f*/);

         float d = sdf.evalDistance(p - 0.2f);
 //        setTrap( sdf.getTrap() );

 //        return  length( make_float3(p.x - 0.5f - 0.5f * sin(m_time / 10.0f)) ); // <- cube forms
         //return  length(p.x-1.0);
 //        return length( make_float3(p.x, p.z, 0.0f) - make_float3(1.0,1.0,0.0f) )-0.05f; // <- tube forms
         return d;
         //return length(p); // <- no trap
     }

    __device__ inline float map(float3 _p)
    {
        float t = m_time / 18.0f;
        t = tan(t);
//        Mandelbulb sdf(m_maxIterations, t);

        float global_t = m_time;
        float3 offset = make_float3(1.0 + 0.2f * cos( 1.0f * (global_t / 5.7f)),
                                    1.0,
                                    0.3 + 0.1f * (cos( 1.0f * (global_t / 1.7f)))
                                    );

        float3 z = _p;
//        float d2 = sdf.evalDistance(z);
//        z.x -= global_t * 0.01f;
//        z = translateHook(0, z);

        // z.x = fmod(z.x, 3.5f);

        z = fabs( make_float3(1.0f) - fmod(z, 2.0f));
//        z.x = fabs(z.x + m_offset.x) - m_offset.x;
//        z.x = fabs(z.x + offset.x) - offset.x;

        const float s = 0.9f;
        float k = 1.0f;
        float m0 = 1e10, m1 = 1e10, m2 = 1e10;

        SphereTrap trapA(1);

        float d = 1000.0f;
        for(int n = 0; n < m_maxIterations; ++n)
        {
            trapA.trap(z);

            m0 = min(m0, dot(z, z) / (k*k) );
            m1 = min(m1, trapA.getTrapValue() );
            m2 = length( make_float3( z.z, z.x, 0.0f) - make_float3(1.0f, 1.0f, 0.0) ) - 0.3f; // <- tube forms
            m2 = remap(m2, 0, 50000, 0.0, 1.0);

            ///@todo rotate
            ///
            // y
            if(z.x + z.y < 0.0){ float3 tmp = z; z.x = -tmp.y; z.y = -tmp.x; }
            z = fabs(z);

            if(z.x + z.z < 0.0){ float3 tmp = z; z.x = -tmp.z; z.z = -tmp.x; }
            z = fabs(z);

            //y
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
         #ifdef MANDELTRAP
             d = min(d, trap(z) * powf(m_scale, -float(n+1)));
         #endif

            k *= s;
        }

        setTrap0( m0 );
        setTrap1( m1 );
        setTrap2( m2 );


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
