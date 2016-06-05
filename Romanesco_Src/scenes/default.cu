// pos 1.78707 0 -1.65409
// rot 0 0.8708 0
// fov 60

#include "romanescocore.h"

HIT_PROGRAM float2 hit(float3 x, int maxIterations, float global_t)
{
	Mandelbulb sdf(maxIterations);
	sdf.evalParameters();
    	sdf.setTime(global_t);
	float p = (5.0 * abs(sin(global_t / 40.0))) + 3.0;
	sdf.setPower( p );

	return make_float2( sdf.evalDistance(x), sdf.getTrap() );
}
