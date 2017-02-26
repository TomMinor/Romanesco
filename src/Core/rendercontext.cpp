#include "include/Core/rendercontext.h"

RenderContext::RenderContext(unsigned int _width, unsigned int _height, unsigned int _samples)
    : m_outputWidth(_width), m_outputHeight(_height), m_samples(_samples)
{
    m_framePath = "frames/out_%04d.exr";
}
