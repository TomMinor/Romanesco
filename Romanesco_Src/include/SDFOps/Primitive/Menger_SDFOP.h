#ifndef MENGERSDFOP_H
#define MENGERSDFOP_H

#include "Base_SDFOP.h"

class MengerSDFOp : public BaseSDFOP
{
public:
    MengerSDFOp();
    ~MengerSDFOp();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
};

#endif // MENGERSDFOP_H
