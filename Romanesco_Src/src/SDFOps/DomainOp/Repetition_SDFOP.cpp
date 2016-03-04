
#include "Repetition_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Repetition_SDFOP::Repetition_SDFOP() :
    BaseSDFOP::BaseSDFOP()
{
    m_returnType = ReturnType::Float;
} 
 
Repetition_SDFOP::~Repetition_SDFOP() 
{ 
} 

std::string Repetition_SDFOP::getFunctionName()
{
    return "repeat";
}

std::string Repetition_SDFOP::getSource()
{

}

Argument Repetition_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Repetition_SDFOP::argumentSize()
{
    return args.size();
}
