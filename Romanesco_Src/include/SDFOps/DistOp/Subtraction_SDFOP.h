
#ifndef SUBTRACTION_SDFOP_H 
#define SUBTRACTION_SDFOP_H 
 
#include "DistOpInterface_SDFOP.h"
 
class Subtraction_SDFOP : public DistOpInterface_SDFOP
{ 
public: 
    Subtraction_SDFOP(); 
    ~Subtraction_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // SUBTRACTION_SDFOP_H 

