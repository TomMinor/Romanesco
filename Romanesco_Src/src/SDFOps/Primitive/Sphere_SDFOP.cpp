#include <sstream>
#include "Sphere_SDFOP.h" 

static const std::vector<Argument> args = {
};

Sphere_SDFOP::Sphere_SDFOP(float _radius) :
    BaseSDFOP::BaseSDFOP(),  m_radius(_radius)
{
    m_returnType = ReturnType::Float;
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
    sourceStream<< "\treturn length(P) - (" << m_radius << ");\n";

    return sourceStream.str();
}

Argument Sphere_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Sphere_SDFOP::argumentSize()
{
    return 0;
}
