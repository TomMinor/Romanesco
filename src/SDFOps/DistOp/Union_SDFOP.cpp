
#include "Union_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};
 
Union_SDFOP::Union_SDFOP()  :
    BaseSDFOP::BaseSDFOP()
{
    m_returnType = ReturnType::Float;
}
 
Union_SDFOP::~Union_SDFOP() 
{
}


std::string Union_SDFOP::getFunctionName()
{
    return "SDF_union";
}

std::string Union_SDFOP::getSource()
{
    return R"(
            return max(-a,b);
)";
}

Argument Union_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Union_SDFOP::argumentSize()
{
    return args.size();
}
