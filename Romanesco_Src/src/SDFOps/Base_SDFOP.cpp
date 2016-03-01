#include "Base_SDFOP.h"

std::set<std::string> BaseSDFOP::m_headers;

static const std::vector<Argument> args = {
        {"default", ReturnType::Float, "0.0f"}
};

BaseSDFOP::BaseSDFOP()
{
    // Everything will probably need this
    m_headers.insert( "cutil_math.h" );
}

BaseSDFOP::~BaseSDFOP()
{

}

unsigned int BaseSDFOP::argumentSize()
{
    return args.size();
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
