
#include "Scale_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Scale_SDFOP::Scale_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Scale_SDFOP::~Scale_SDFOP() 
{ 
} 


std::string Scale_SDFOP::getFunctionName()
{
    return "scale";
}

std::string Scale_SDFOP::getSource()
{

}

Argument Scale_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
