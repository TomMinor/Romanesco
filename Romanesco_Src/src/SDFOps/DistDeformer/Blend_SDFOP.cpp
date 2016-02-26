#include "DistDeformer/Blend_SDFOP.h"
 
Blend_SDFOP::Blend_SDFOP() 
{ 
} 
 
Blend_SDFOP::~Blend_SDFOP() 
{ 
} 

std::string Blend_SDFOP::getFunctionName()
{
    return "blend";
}

std::string Blend_SDFOP::getSource()
{

}

std::string Blend_SDFOP::getDefaultArg(unsigned int index)
{
    // Base implementation does error checking, we discard any result
    BaseSDFOP::getDefaultArg(index);

    static const std::vector<std::string> args = { "a", "b" };

    return args[index];
}
