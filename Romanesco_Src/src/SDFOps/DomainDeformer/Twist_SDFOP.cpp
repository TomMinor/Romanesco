
#include "Twist_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Twist_SDFOP::Twist_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Twist_SDFOP::~Twist_SDFOP() 
{ 
} 


std::string Twist_SDFOP::getFunctionName()
{
    return "twist";
}

std::string Twist_SDFOP::getSource()
{

}

Argument Twist_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Twist_SDFOP::argumentSize()
{
    return args.size();
}
