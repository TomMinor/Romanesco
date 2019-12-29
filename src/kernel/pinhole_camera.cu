
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_world.h>
#include <optix_device.h>
#include "helpers.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float3 result_nrm;
  float3 result_world;
  float result_depth;
  float  importance;
  int depth;
};

rtDeclareVariable(Matrix4x4, normalmatrix, , );
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );

rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              output_buffer_nrm;
rtBuffer<float4, 2>              output_buffer_depth;
rtBuffer<float4, 2>              output_buffer_world;

rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

//#define TIME_VIEW

RT_PROGRAM void pinhole_camera()
{
#ifdef TIME_VIEW
  clock_t t0 = clock(); 
#endif
  float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = (d.x*U + d.y*V + W);
  
  ray_direction = make_float3((make_float4(ray_direction, 1.0) * normalmatrix));
  ray_direction = normalize(ray_direction);

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

#ifdef TIME_VIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
  //output_buffer[launch_index] = make_float4(  pixel_time ) ); 
  output_buffer[launch_index] = make_float4( prd.result, 1.0 );
  output_buffer_nrm[launch_index] = make_float4( prd.result_nrm, 1.0 );
  output_buffer_depth[launch_index] = make_float4( prd.result_depth, prd.result_depth, prd.result_depth,1.0 );
  output_buffer_world[launch_index] = make_float4( prd.result_world, 1.0 );
#else
  output_buffer[launch_index] = make_float4( prd.result, 1.0 );
  output_buffer_nrm[launch_index] = make_float4( prd.result_nrm, 1.0 );
  output_buffer_depth[launch_index] = make_float4( prd.result_depth, prd.result_depth, prd.result_depth,1.0 );
  output_buffer_world[launch_index] = make_float4( prd.result_world, 1.0 );
#endif
}

RT_PROGRAM void env_camera()
{
  size_t2 screen = output_buffer.size();

  float2 d = make_float2(launch_index) / make_float2(screen) * make_float2(2.0f * M_PIf , M_PIf) + make_float2(M_PIf, 0);
  float3 angle = make_float3(cos(d.x) * sin(d.y), -cos(d.y), sin(d.x) * sin(d.y));
  float3 ray_origin = eye;
  float3 ray_direction = normalize(angle.x*normalize(U) + angle.y*normalize(V) + angle.z*normalize(W));

  ray_direction = make_float3((make_float4(ray_direction, 1.0) * normalmatrix));
  ray_direction = normalize(ray_direction);

  optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  output_buffer[launch_index] = make_float4( prd.result );
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_float4( bad_color, 1.0 );
}
