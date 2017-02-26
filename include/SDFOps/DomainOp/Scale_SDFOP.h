
#ifndef SCALE_SDFOP_H 
#define SCALE_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Scale_SDFOP : public BaseSDFOP 
{ 
public: 
    Scale_SDFOP(); 
    ~Scale_SDFOP(); 

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // SCALE_SDFOP_H 

