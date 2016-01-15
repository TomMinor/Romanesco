/*
 * Using CUDA 7.0 NVRTC with Thrust.
 *
 * Copyright 2015 Applied Parallel Computing LLC.
 * http://parallel-computing.pro
 *
 * All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to intellectual property rights under U.S. and international
 * Copyright laws.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, APPLIED PARALLEL COMPUTING LLC MAKES NO 
 * REPRESENTATION ABOUT THE SUITABILITY OF THESE LICENSED DELIVERABLES
 * FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND. APPLIED PARALLEL COMPUTING LLC DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THESE LICENSED DELIVERABLES, INCLUDING ALL
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS
 * FOR A PARTICULAR PURPOSE. NOTWITHSTANDING ANY TERMS OR CONDITIONS TO
 * THE CONTRARY IN THE LICENSE AGREEMENT, IN NO EVENT SHALL APPLIED
 * PARALLEL COMPUTING LLC BE LIABLE FOR ANY SPECIAL, INDIRECT,
 * INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER
 * RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 * CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THESE LICENSED DELIVERABLES.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include the above Disclaimer in the user documentation
 * and internal comments to the code.
 */

// #include <cstdio>
// #include <thrust/for_each.h>
// #include <thrust/device_vector.h>

__device__ int function(int x, int y, int z);

// struct functor
// {
// 	__device__ void operator()(int x)
// 	{
// 		// int result = function(x, 0);
// 		// printf("Result = %d\n", result);
// 	}
// };

__global__ void vecInit(float *A,int vecsize) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < vecsize) {
        A[idx]=idx;
    }
}

__global__ void vecAdd(float *A, float *B, float *C, int vecsize) {
    // int idx = threadIdx.x + blockDim.x * blockIdx.x;

 //    functor tmp;

	// tmp(1);
	// tmp(5);
	// tmp(9);

    int result = function(threadIdx.x, blockDim.x, blockIdx.x);
}


// void for_each()
// {
// 	printf("for_each\n");



// 	// thrust::device_vector<int> d_vec(3);
// 	// d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;
// 	// thrust::for_each(d_vec.begin(), d_vec.end(), functor());
// }

