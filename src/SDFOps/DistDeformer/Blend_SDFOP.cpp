#include "Blend_SDFOP.h"
 
static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Blend_SDFOP::Blend_SDFOP()
{
	//BaseSDFOP();
} 
 
Blend_SDFOP::~Blend_SDFOP() 
{ 
} 

std::string Blend_SDFOP::getFunctionName()
{
    return "blend";
}

std::string Blend_SDFOP::getSource()
{
	return R"(
            return 0.0f;
)";
}

Argument Blend_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Blend_SDFOP::argumentSize()
{
    return args.size();
}
