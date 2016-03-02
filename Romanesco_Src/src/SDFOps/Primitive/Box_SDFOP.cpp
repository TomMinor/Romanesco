
#include "Box_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Box_SDFOP::Box_SDFOP() 
{
    m_returnType = ReturnType::Float;
} 
 
Box_SDFOP::~Box_SDFOP() 
{ 
} 

std::string Box_SDFOP::getFunctionName()
{
    return "box";
}

std::string Box_SDFOP::getSource()
{

}

Argument Box_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
