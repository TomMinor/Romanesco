// pos 1.62512 0 0.00249628
// rot 0 10.9708 0
// fov 60


#include "romanescocore.h"
#include "tunneltest.h"

HIT_PROGRAM float2 hit(float3 x, int maxIterations, float global_t)
{
	TunnelTest sdf(maxIterations);
	sdf.evalParameters();
    	sdf.setTime(global_t);
    	sdf.setTranslateHook( 0, x - make_float3(-global_t, 0.0f, 0.0f) );

	return make_float2( sdf.evalDistance(x), sdf.getTrap() );
}
