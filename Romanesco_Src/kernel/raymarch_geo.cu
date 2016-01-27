#include <vector_types.h>
#include "cutil_math.h"

// Disable name mangling
extern "C"
{

// Default implementation
__device__ __attribute__ ((noinline)) float distancehit_hook(float3 p, float iteration)
{
    return -1.0f;
}

}
