#include <sstream>
#include "Transform_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Mat4, "0.0f"}
};

Transform_SDFOP::Transform_SDFOP(const glm::vec3 &_m) :
    BaseSDFOP::BaseSDFOP(), m_transform(_m)
{
    m_returnType = ReturnType::Float;
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



std::string Transform_SDFOP::getFunctionName()
{
    return "transform";
}


Argument Transform_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Transform_SDFOP::argumentSize()
{
    return args.size();
}
