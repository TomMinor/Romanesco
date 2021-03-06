
#include "Capsule_SDFOP.h" 

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Capsule_SDFOP::Capsule_SDFOP()//  :
   // BaseSDFOP::BaseSDFOP()
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
	return R"(
            return 0.0f;
)";
}

Argument Capsule_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Capsule_SDFOP::argumentSize()
{
    return args.size();
}
