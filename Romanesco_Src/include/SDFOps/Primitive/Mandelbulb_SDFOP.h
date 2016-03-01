#ifndef MANDELBULB_SDFOP_H
#define MANDELBULB_SDFOP_H

#include "Base_SDFOP.h"

class Mandelbulb_SDFOP : public BaseSDFOP
{
public:
    Mandelbulb_SDFOP();
    ~Mandelbulb_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
};

#endif // MANDELBULB_SDFOP_H
