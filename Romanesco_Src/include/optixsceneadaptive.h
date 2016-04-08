#ifndef OPTIXSCENEADAPTIVE_H
#define OPTIXSCENEADAPTIVE_H

#include "optixscene.h"

class OptixSceneAdaptive : public OptixScene
{
    enum class CameraType
    {
        Pinhole,
        AdaptivePinhole
    };

public:
    OptixSceneAdaptive(unsigned int _width, unsigned int _height);
    ~OptixSceneAdaptive();

private:
    CameraType m_cameraType;

};

#endif
