#ifndef RENDERCONTEXT_H
#define RENDERCONTEXT_H

#include <string>

struct RenderContext
{
    RenderContext(unsigned int _width, unsigned int _height, unsigned int _samples = 32);

    unsigned int m_outputWidth;
    unsigned int m_outputHeight;

    unsigned int m_samples;

    std::string m_framePath;
};

#endif // RENDERCONTEXT_H
