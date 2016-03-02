
#ifndef DISPLACE_SDFOP_H 
#define DISPLACE_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Displace_SDFOP : public BaseSDFOP 
{ 
public: 
    Displace_SDFOP(); 
    ~Displace_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // DISPLACE_SDFOP_H 

