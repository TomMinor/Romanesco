#include <sstream>
#include "Sphere_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Sphere_SDFOP::Sphere_SDFOP(float _radius)
    : m_radius(_radius)
{

} 
 
Sphere_SDFOP::~Sphere_SDFOP() 
{

} 


std::string Sphere_SDFOP::getFunctionName()
{
    return "sphere";
}

std::string Sphere_SDFOP::getSource()
{
    std::ostringstream sourceStream;
    sourceStream<< "return length(_p) - (" << m_radius << ");\n";

    return sourceStream.str();
}

Argument Sphere_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
