// pos 3.525 0 7.33048e-06
// rot 0 1.5707 0
// fov 60


#include "romanescocore.h"
#include "tunneltest.h"

HIT_PROGRAM float4 hit(float3 x, int maxIterations, float global_t)
{
	TunnelTest sdf(maxIterations);
	sdf.evalParameters();
    	sdf.setTime(global_t);
    	sdf.setTranslateHook( 0, x - make_float3(-global_t, 0.0f, 0.0f) );

	return make_float4( sdf.evalDistance(x), 
						sdf.getTrap0(), 
						sdf.getTrap1(), 
						sdf.getTrap2() );
}
