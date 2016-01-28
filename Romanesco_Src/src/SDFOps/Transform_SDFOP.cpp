#include <sstream>
#include "Transform_SDFOP.h" 
 
Transform_SDFOP::Transform_SDFOP(const glm::vec3 &_m)
    : m_transform(_m)
{

} 
 
Transform_SDFOP::~Transform_SDFOP() 
{

}


std::string Transform_SDFOP::getSource()
{
    std::ostringstream sourceStream;
    sourceStream << "_p += make_float3(" << m_transform.x << "," << m_transform.y << "," << m_transform.z << ");\n";

    return sourceStream.str();
}


