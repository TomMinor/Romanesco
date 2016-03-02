#include "Base_SDFOP.h"

std::set<std::string> BaseSDFOP::m_headers;

static const std::vector<Argument> args = {
        {"default", ReturnType::Float, "0.0f"}
};

BaseSDFOP::BaseSDFOP()
{
    // Everything will probably need this
    m_headers.insert( "cutil_math.h" );
    m_returnType = ReturnType::Void;
}

BaseSDFOP::~BaseSDFOP()
{

}

std::string BaseSDFOP::getTypeString()
{
    switch(m_returnType)
    {
    case ReturnType::Void:
        return "void";
        break;
    case ReturnType::Float:
        return "float";
        break;
    case ReturnType::Int:
        return "int";
        break;
    case ReturnType::Vec3:
        return "vec3";
        break;
    case ReturnType::Mat4:
        return "mat4";
        break;
    default:
        throw std::runtime_error("Invalid type string requested");
    }
}

unsigned int BaseSDFOP::argumentSize()
{
    return args.size() - 1;
}

std::string BaseSDFOP::getSource()
{
    return "assert(\"UNUSED\")";
}

std::string BaseSDFOP::getFunctionName()
{
    return "undefined";
}

Argument BaseSDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
