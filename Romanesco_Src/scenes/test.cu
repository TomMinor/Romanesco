// pos 3.075 0 5.70148e-06
// rot 0 1.5708 0
// fov 60

#include "romanescocore.h"

HIT_PROGRAM float hit(float3 x, uint maxIterations, float global_t)
{
	Mandelbulb sdf(maxIterations);
	sdf.evalParameters();
    	sdf.setTime(global_t);

	return sdf.evalDistance(x);
}
