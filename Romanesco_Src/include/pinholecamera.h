#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include <optix.h>
#include <sutil.h>
#include <optixu/optixu.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>

class MyPinholeCamera {
  typedef optix::float3 float3;
  typedef optix::float2 float2;
public:
  enum AspectRatioMode {
    KeepVertical,
    KeepHorizontal,
    KeepNone
  };

  MyPinholeCamera(float3 eye, float3 lookat, float3 up, float hfov=60, float vfov=60,
                         AspectRatioMode arm = KeepVertical);

  void setup();

  void getEyeUVW(float3& eye, float3& U, float3& V, float3& W);

  void getEyeLookUpFOV(float3& eye, float3& lookat, float3& up, float& HFOV, float& VFOV);

  void scaleFOV(float);
  void translate(float2);
  void dolly(float);
  void transform( const optix::Matrix4x4& trans );
  void setAspectRatio(float ratio);

  void setParameters(float3 eye_in, float3 lookat_in, float3 up_in, float hfov_in, float vfov_in, MyPinholeCamera::AspectRatioMode aspectRatioMode_in);

  enum TransformCenter {
    LookAt,
    Eye,
    Origin
  };

  float3 eye, lookat, up;
  float hfov, vfov;
private:
  float3 lookdir, camera_u, camera_v;
  AspectRatioMode aspectRatioMode;
};
#endif // PINHOLECAMERA_H
