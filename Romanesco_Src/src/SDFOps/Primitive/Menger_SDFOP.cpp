#include "Menger_SDFOP.h"

static const std::vector<Argument> args = {
    {"a", ReturnType::Float, "0.0f"},
    {"b", ReturnType::Float, "0.0f"}
};

MengerSDFOp::MengerSDFOp()
{
m_returnType = ReturnType::Float;
}

MengerSDFOp::~MengerSDFOp()
{

}

std::string MengerSDFOp::getFunctionName()
{
    return "mengersponge";
}

std::string MengerSDFOp::getSource()
{

}

Argument MengerSDFOp::getArgument(unsigned int index)
{
    return args.at(index);
}
