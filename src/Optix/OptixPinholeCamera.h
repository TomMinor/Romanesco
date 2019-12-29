#ifndef PINHOLECAMERA_H
#define PINHOLECAMERA_H

#include "PinholeCamera.h"
#include "OptixHeaders.h"

namespace Romanesco {

///
/// \brief The PinholeCamera class modified from the one provided in the Optix demos
///
class OptixPinholeCamera : public PinholeCamera {
	public:
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

		optix::float3 eye, lookat, up;
		float hfov, vfov;
};

};
#endif // PINHOLECAMERA_H
