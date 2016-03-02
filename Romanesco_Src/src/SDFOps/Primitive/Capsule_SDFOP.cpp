
#include "Capsule_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Capsule_SDFOP::Capsule_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Capsule_SDFOP::~Capsule_SDFOP() 
{ 
} 

std::string Capsule_SDFOP::getFunctionName()
{
    return "capsule";
}

std::string Capsule_SDFOP::getSource()
{

}

Argument Capsule_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
