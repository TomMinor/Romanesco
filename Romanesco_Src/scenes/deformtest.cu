// pos 3.525 0 7.33048e-06
// rot 0 1.5708 0
// fov 60

#include "romanescocore.h"
// #define MANDELTRAP
#include "tunneltest.h"

HIT_PROGRAM float4 hit(float3 x, int maxIterations, float global_t)
{
	float3 X = x;
	// X -= make_float3(-global_t / 4.5f + 8.0f, 0.0f, 0.0f);

	TunnelTest sdf(maxIterations);
    sdf.evalParameters();
    sdf.setTime(global_t);
    sdf.setTranslateHook( 0, make_float3(-global_t / 4.5f, 0.0f, 0.0f) );

    float3 particle = make_float3(3.525f, 0.0f, 0.0f);
    const float part_dist = length( particle - X );
    const float force = smoothstep( 0.0f, 1.0f, 0.1f / (part_dist*part_dist) ) * 4.0f;
    const float3 weg = (X - particle) / max(0.01f, part_dist);

    X -= weg * force;
    float d1 = sdf.evalDistance(X);
    
    float d = d1;

	return make_float4( d, 
						sdf.getTrap0(), 
						sdf.getTrap1(), 
						sdf.getTrap2() );
}
