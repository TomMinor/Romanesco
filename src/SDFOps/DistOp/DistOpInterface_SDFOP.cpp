#include "include/SDFOps/DistOp/DistOpInterface_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

DistOpInterface_SDFOP::DistOpInterface_SDFOP()
{
}

DistOpInterface_SDFOP::~DistOpInterface_SDFOP()
{
}


std::string DistOpInterface_SDFOP::getFunctionName()
{
    return "union";
}

std::string DistOpInterface_SDFOP::getSource()
{

}

Argument DistOpInterface_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int DistOpInterface_SDFOP::argumentSize()
{
    return args.size();
}
