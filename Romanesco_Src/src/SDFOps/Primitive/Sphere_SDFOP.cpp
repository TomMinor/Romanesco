#include <sstream>
#include "Sphere_SDFOP.h" 
 
Sphere_SDFOP::Sphere_SDFOP(float _radius)
    : m_radius(_radius)
{

} 
 
Sphere_SDFOP::~Sphere_SDFOP() 
{

} 

std::string Sphere_SDFOP::getSource()
{
    std::ostringstream sourceStream;
    sourceStream<< "return length(_p) - (" << m_radius << ");\n";

    return sourceStream.str();
}
