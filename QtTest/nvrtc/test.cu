#include <thrust/for_each.h>
#include <thrust/device_vector.h>

__device__ void function(int x);

struct functor
{
    __device__ void operator()(int x)
    {
        function(x);
    }
};

void for_each()
{
    printf("for_each\n");

    thrust::device_vector d_vec(3);
    d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;
    thrust::for_each(d_vec.begin(), d_vec.end(), functor());
}
