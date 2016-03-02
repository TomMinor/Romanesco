
#ifndef TWIST_SDFOP_H 
#define TWIST_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Twist_SDFOP : public BaseSDFOP 
{ 
public: 
    Twist_SDFOP(); 
    ~Twist_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
}; 
 
#endif // TWIST_SDFOP_H 

