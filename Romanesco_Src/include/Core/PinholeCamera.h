#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include <optix.h>
#include <sutil.h>
#include <optixu/optixu.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>

///
/// \brief The PinholeCamera class modified from the one provided in the Optix demos
///
class PinholeCamera {

public:
  enum AspectRatioMode {
    KeepVertical,
    KeepHorizontal,
    KeepNone
  };

  PinholeCamera(optix::float3 eye, optix::float3 lookat, optix::float3 up, float hfov=60, float vfov=60, AspectRatioMode arm = KeepVertical);

  void setup();

  void getEyeUVW(optix::float3& eye, optix::float3& U, optix::float3& V, optix::float3& W);

  void getEyeLookUpFOV(optix::float3& eye, optix::float3& lookat, optix::float3& up, float& HFOV, float& VFOV);

  void scaleFOV(float);
  void translate(optix::float2);
  void dolly(float);
  void transform( const optix::Matrix4x4& trans );
  void setAspectRatio(float ratio);

  void setParameters(optix::float3 eye_in, optix::float3 lookat_in, optix::float3 up_in, float hfov_in, float vfov_in, PinholeCamera::AspectRatioMode aspectRatioMode_in);

  enum TransformCenter {
    LookAt,
    Eye,
    Origin
  };

  optix::float3 eye, lookat, up;
  float hfov, vfov;
private:
  optix::float3 lookdir, camera_u, camera_v;
  AspectRatioMode aspectRatioMode;
};
#endif // PINHOLECAMERA_H
