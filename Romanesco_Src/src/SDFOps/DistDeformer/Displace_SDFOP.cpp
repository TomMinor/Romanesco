
#include "Displace_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"},
};
 
Displace_SDFOP::Displace_SDFOP() 
{ 
} 
 
Displace_SDFOP::~Displace_SDFOP() 
{ 
} 

std::string Displace_SDFOP::getFunctionName()
{
    return "displace";
}

std::string Displace_SDFOP::getSource()
{
    return
    R"("
    {
        void arse;
    }
    ")";
}

Argument Displace_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}
