
#include "Subtraction_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};
 
Subtraction_SDFOP::Subtraction_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Subtraction_SDFOP::~Subtraction_SDFOP() 
{ 
}

std::string Subtraction_SDFOP::getFunctionName()
{
    return "subtract";
}

std::string Subtraction_SDFOP::getSource()
{

}

Argument Subtraction_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Subtraction_SDFOP::argumentSize()
{
    return args.size();
}
