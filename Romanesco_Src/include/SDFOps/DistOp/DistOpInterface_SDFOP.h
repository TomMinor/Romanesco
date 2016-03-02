#ifndef DISTOPINTERFACE_SDFOPH_H
#define DISTOPINTERFACE_SDFOPH_H

#include "Base_SDFOP.h"

class DistOpInterface_SDFOP : public BaseSDFOP
{
public:
    DistOpInterface_SDFOP();
    ~DistOpInterface_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
};

#endif // DISTOPINTERFACE_SDFOPH_H
