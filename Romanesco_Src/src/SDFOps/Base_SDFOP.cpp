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
