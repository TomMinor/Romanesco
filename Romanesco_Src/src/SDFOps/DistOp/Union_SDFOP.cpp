
#include "Union_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};
 
Union_SDFOP::Union_SDFOP() 
{ 
} 
 
Union_SDFOP::~Union_SDFOP() 
{
}


std::string Union_SDFOP::getFunctionName()
{
    return "union";
}

std::string Union_SDFOP::getSource()
{

}

Argument Union_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
