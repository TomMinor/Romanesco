#include "Mandelbulb_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

Mandelbulb_SDFOP::Mandelbulb_SDFOP() :
    BaseSDFOP::BaseSDFOP()
{
m_returnType = ReturnType::Float;
}

Mandelbulb_SDFOP::~Mandelbulb_SDFOP()
{

}

std::string Mandelbulb_SDFOP::getFunctionName()
{
    return "mandelbulb";
}

std::string Mandelbulb_SDFOP::getSource()
{

}

Argument Mandelbulb_SDFOP::getArgument(unsigned int index)
{
    return args.at(index);
}

unsigned int Mandelbulb_SDFOP::argumentSize()
{
    return args.size();
}
