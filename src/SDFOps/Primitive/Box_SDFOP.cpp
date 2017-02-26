#include <sstream>
#include "Box_SDFOP.h" 

static const std::vector<Argument> args = {
};

Box_SDFOP::Box_SDFOP()  :
    BaseSDFOP::BaseSDFOP()
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
    std::ostringstream sourceStream;
    sourceStream<< "\treturn length( make_float3( max(vars.P.x - 0.7f, 0.0f), max(vars.P.y - 0.7f, 0.0f), max(vars.P.z - 0.7f, 0.0f)  ) );";

    return sourceStream.str();
}

Argument Box_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Box_SDFOP::argumentSize()
{
    return 0;
}
