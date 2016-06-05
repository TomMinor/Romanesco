// pos -2.97333 0 -1.96865
// rot 0 -1.5292 0
// fov 30

#include "romanescocore.h"

HIT_PROGRAM float2 hit(float3 x, int maxIterations, float global_t)
{
	x.z += (global_t / 25.0f);
	Matrix4x4 transform = Matrix4x4::rotate( global_t / 25.0f, make_float3(0,0,1) );
	x = applyTransform(x, transform);

	Mandelbulb sdf(maxIterations);
	sdf.setTime(global_t);
	sdf.evalParameters();
	
	float oscillatingTime = sin( global_t / 300.0f );
	float p = (5.0f * oscillatingTime) + 3.0f;
	sdf.setPower(p);

	return make_float2( sdf.evalDistance(x), sdf.getTrap() );
}
