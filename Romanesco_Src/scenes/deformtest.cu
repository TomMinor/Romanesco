// pos 3.525 0 7.33048e-06
// rot 0 1.5708 0
// fov 60

#include "romanescocore.h"
#define MANDELTRAP
#include "tunneltest.h"

HIT_PROGRAM float4 hit(float3 x, int maxIterations, float global_t)
{
    float3 X = x;
    X += make_float3(-global_t / 6.5f + 4.0f, 0.0f, 0.0f);

    TunnelTest sdf(maxIterations);
    sdf.evalParameters();
    sdf.setTime(global_t);
    sdf.setTranslateHook( 0, make_float3(-global_t / 4.5f, 0.0f, 0.0f) );

    Mandelbulb sdf_bulb(maxIterations);
    sdf_bulb.evalParameters();	
    sdf_bulb.setTime(global_t);
    //sdf_bulb.setTranslateHook( 0, make_float3(-global_t, 0.0f, 0.0f) );
    sdf_bulb.setRotateHook( 0, make_float3( radians(global_t) ) );
    sdf_bulb.setPower(5.0f);

    float3 particle = make_float3(3.525f, 0.0f, 0.0f);
    const float part_dist = length( particle - X );
    const float force = smoothstep( 0.0f, 1.0f, 0.05f / (part_dist*part_dist) ) * 12.0f;
    const float3 weg = (X - particle) / max(0.01f, part_dist);

    X -= weg * force;
    float d1 = sdf.evalDistance(X);
    float d2 = sdf_bulb.evalDistance(X);
    X.y -= 1;
    float d3 = sdBox(X  + make_float3(4.0f, 0.0f, 0.0f), make_float3(8.0f, 2.0f, 2.0f) );
    float d4 = sdSphere(X  + make_float3(1.5f, 0.0f, 0.0f), 10.0f );
    
    float d = max(d1,d3);
//    d = max(d, -d2);
    d = max(d, d4);
    d = max(d, d2);

    return make_float4( d, 
                        sdf.getTrap0(), 
                        sdf.getTrap1(), 
                        sdf.getTrap2() );
}
