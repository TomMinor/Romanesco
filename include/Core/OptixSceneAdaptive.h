#ifndef OPTIXSCENEADAPTIVE_H
#define OPTIXSCENEADAPTIVE_H

#include <QOpenGLFunctions_4_3_Core>

#include "OptixScene.h"

class OptixSceneAdaptive : public OptixScene
{
    enum class CameraType
    {
        Pinhole,
        AdaptivePinhole
    };

public:
    OptixSceneAdaptive(unsigned int _width, unsigned int _height, QOpenGLFunctions_4_3_Core* _gl);
    ~OptixSceneAdaptive();

private:
    CameraType m_cameraType;

};

#endif
