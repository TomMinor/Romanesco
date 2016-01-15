#include <vector_types.h>
#include "cutil_math.h"

extern "C"
{

__device__ float3 shade(float3 p, float iteration)
{
    return make_float3(0,1,0);
}

}
