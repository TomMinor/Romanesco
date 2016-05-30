// pos 3.075 0 5.70148e-06
// rot 0 1.5708 0
// fov 60

#include "romanescocore.h"

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
        float a = map(_p) * m_fudgeFactor;
        _p.y += 1;
        float b = sdBox(_p  - make_float3(-4.0f, 0.0f, 0.0f), m_limits);
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
        translateHook(0, z);

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

HIT_PROGRAM float hit(float3 x, uint maxIterations, float global_t)
{
	TunnelTest sdf(maxIterations);
	sdf.evalParameters();
    	sdf.setTime(global_t);

	return sdf.evalDistance(x);
}