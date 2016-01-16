#include <vector_types.h>
#include "cutil_math.h"

// Disable name mangling
extern "C"
{

// Default implementation
__device__ __attribute__ ((noinline)) float3 shade_hook(float3 p, float3 nrm, float iteration)
{
    return make_float3(1, 0, 0);
}

}

