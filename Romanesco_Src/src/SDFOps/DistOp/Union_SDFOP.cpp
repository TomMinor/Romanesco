
#include "Union_SDFOP.h" 
 
Union_SDFOP::Union_SDFOP() 
{ 
} 
 
Union_SDFOP::~Union_SDFOP() 
{ 
} 

std::string Union_SDFOP::getDefaultArg(unsigned int index)
{
    // Base implementation does error checking, we discard any result
    BaseSDFOP::getDefaultArg(index);

    static const std::vector<std::string> args = { "a", "b" };

    return args[index];
}
