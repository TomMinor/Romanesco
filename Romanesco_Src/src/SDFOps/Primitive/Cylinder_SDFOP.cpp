
#include "Cylinder_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Cylinder_SDFOP::Cylinder_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Cylinder_SDFOP::~Cylinder_SDFOP() 
{ 
} 

std::string Cylinder_SDFOP::getFunctionName()
{
    return "cylinder";
}

std::string Cylinder_SDFOP::getSource()
{

}

Argument Cylinder_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
