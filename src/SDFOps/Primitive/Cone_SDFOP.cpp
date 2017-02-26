
#include "Cone_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Cone_SDFOP::Cone_SDFOP()  :
    BaseSDFOP::BaseSDFOP()
{
    m_returnType = ReturnType::Float;
} 
 
Cone_SDFOP::~Cone_SDFOP() 
{ 
} 

std::string Cone_SDFOP::getFunctionName()
{
    return "cone";
}

std::string Cone_SDFOP::getSource()
{

}

Argument Cone_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Cone_SDFOP::argumentSize()
{
    return args.size();
}
