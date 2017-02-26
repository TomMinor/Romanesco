// pos 0.729305 -0.0823601 2.72943
// rot 1.49012e-08 -3.3292 0
// fov 60


#include "romanescocore.h"
//#define MANDELTRAP
#include "tunneltest.h"

HIT_PROGRAM float4 hit(float3 x, int maxIterations, float global_t)
{
	TunnelTest sdf(maxIterations);
	sdf.evalParameters();
	sdf.setTime(global_t);
	//sdf.setTranslateHook( 0, make_float3(-global_t / 4.5f, 0.0f, 0.0f) );

	return make_float4( sdf.evalDistance(x), 
						sdf.getTrap0(), 
						sdf.getTrap1(), 
						sdf.getTrap2() );
}
