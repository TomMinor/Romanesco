// pos 3.075 0 5.70148e-06
// rot 0 1.5708 0
// fov 60

#include "romanescocore.h"
#include "tunneltest.h"

HIT_PROGRAM float hit(float3 x, uint maxIterations, float global_t)
{
	TunnelTest sdf(maxIterations);
	sdf.evalParameters();
    	sdf.setTime(global_t);
    	sdf.x -= global_t;
    	sdf.setTranslateHook( 0, x );

	return sdf.evalDistance(x);
}
