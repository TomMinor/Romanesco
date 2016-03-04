
#include "Intersection_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};
 
Intersection_SDFOP::Intersection_SDFOP() :
    BaseSDFOP::BaseSDFOP()
{ 
    m_returnType = ReturnType::Float;
} 
 
Intersection_SDFOP::~Intersection_SDFOP() 
{ 
} 

std::string Intersection_SDFOP::getFunctionName()
{
    return "intersect";
}

std::string Intersection_SDFOP::getSource()
{

}

Argument Intersection_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Intersection_SDFOP::argumentSize()
{
    return args.size();
}
