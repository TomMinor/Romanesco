#pragma once

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

namespace Romanesco {

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

	PinholeCamera(glm::vec3 eye, glm::vec3 lookat, glm::vec3 up, float hfov=60, float vfov=60, AspectRatioMode arm = KeepVertical);

	void setup();

	void getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W);

	void getEyeLookUpFOV(glm::vec3& eye, glm::vec3& lookat, glm::vec3& up, float& HFOV, float& VFOV);

	void scaleFOV(float);
	void translate(glm::vec2);
	void dolly(float);
	void transform( const glm::mat4x4& trans );
	void setAspectRatio(float ratio);

	void setParameters(glm::vec3 eye_in, glm::vec3 lookat_in, glm::vec3 up_in, float hfov_in, float vfov_in, PinholeCamera::AspectRatioMode aspectRatioMode_in);

	enum TransformCenter {
		LookAt,
		Eye,
		Origin
	};

	glm::vec3 eye, lookat, up;
	float hfov, vfov;

private:
	glm::vec3 lookdir, camera_u, camera_v;
	AspectRatioMode aspectRatioMode;
};

};