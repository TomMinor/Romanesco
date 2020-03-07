#pragma once

#include <glm/vec3.hpp>
#include <glm/gtx/quaternion.hpp>
#include <qobject.h>


struct GlobalRenderOptions {
	unsigned int width;
	unsigned int height;
	unsigned short samplesPerPixel;

	unsigned short maxIterations;
	double normalDelta;
	double surfaceEpsilon;
};

struct CameraOptions {
	enum class CameraProjection : unsigned int {
		PINHOLE = 0u,
		ENVIRONMENT = 1u,

		TOTALCAMERATYPES
	};

	glm::vec3 cameraEye;
	glm::quat cameraDir;
	CameraProjection cameraProj;

	float fov;
};

struct LocalRenderOptions {
	double time;

	CameraOptions camera;
};

class BaseScene : public QObject {
	Q_OBJECT

public:
	enum class PathTraceRay : unsigned int {
		CAMERA = 0u,
		SHADOW = 1u,
		BSDF = 2u
	};

	BaseScene(GlobalRenderOptions, QOpenGLFunctions_4_3_Core* _gl, QObject *_parent = 0);
	virtual ~BaseScene();

	virtual void drawToBuffer();

	virtual void initialiseScene();
	virtual void createCameras();
	virtual void createWorld();
	virtual void createBuffers();
	virtual void createLights();
	virtual void createLightGeo();

	virtual void setGeometryHitProgram(std::string _hit_src);
	virtual void setShadingProgram(std::string _hit_src);
	virtual void setOutputBuffer(std::string _name);
	std::string outputBuffer();

	void setWidth(int w);
	void setHeight(int h);

	float fov() const;
	int width() const;
	int height() const;
	float time() const;

	///
	/// \brief getBufferContents
	/// \param _name
	/// \return a copy of the buffer contents or null if the buffer doesn't exist
	///
	float* getBufferContents(
		std::string name,
		unsigned int& elementSizeOut,
		unsigned int& widthOut,
		unsigned int& heightOut
	) const;

	bool saveBuffersToDisk(std::string _filename) const;


protected:
	virtual void updateCamera() = 0;
	virtual void updateBufferSize(unsigned int _width, unsigned int _height) = 0;

private:
	float m_fov;
	int m_width;
	int m_height;

};