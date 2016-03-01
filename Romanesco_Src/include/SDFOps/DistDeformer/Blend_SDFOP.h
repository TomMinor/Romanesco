
#ifndef BLEND_SDFOP_H 
#define BLEND_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Blend_SDFOP : public BaseSDFOP 
{ 
public: 
    Blend_SDFOP();
    ~Blend_SDFOP(); 

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
}; 
 
#endif // BLEND_SDFOP_H 

