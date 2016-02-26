#include "Base_SDFOP.h"

std::set<std::string> BaseSDFOP::m_headers;

BaseSDFOP::BaseSDFOP()
{
    // Everything will probably need this
    m_headers.insert( "cutil_math.h" );
}

BaseSDFOP::~BaseSDFOP()
{

}

std::string BaseSDFOP::getSource()
{
    return "assert(\"UNUSED\")";
}

std::string BaseSDFOP::getFunctionName()
{
    return "undefined";
}

std::string BaseSDFOP::getDefaultArg(unsigned int index)
{
    static const std::vector<std::string> args = { "default" };

    if(index > (args.size() - 1) )
    {
        throw std::out_of_range("Default argument index out of range");
    }

    return args[index];
}
