
#include "Bend_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};
 
Bend_SDFOP::Bend_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Bend_SDFOP::~Bend_SDFOP() 
{ 
}


std::string Bend_SDFOP::getFunctionName()
{
    return "bend";
}

std::string Bend_SDFOP::getSource()
{

}

Argument Bend_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}


