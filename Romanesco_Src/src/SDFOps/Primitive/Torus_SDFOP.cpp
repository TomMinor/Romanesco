
#include "Torus_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Torus_SDFOP::Torus_SDFOP()  :
    BaseSDFOP::BaseSDFOP()
{ 
    m_returnType = ReturnType::Float;
} 
 
Torus_SDFOP::~Torus_SDFOP() 
{ 
} 

std::string Torus_SDFOP::getFunctionName()
{
    return "torus";
}

std::string Torus_SDFOP::getSource()
{

}

Argument Torus_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}


unsigned int Torus_SDFOP::argumentSize()
{
    return args.size();
}
