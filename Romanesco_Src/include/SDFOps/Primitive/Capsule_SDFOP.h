
#ifndef CAPSULE_SDFOP_H 
#define CAPSULE_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Capsule_SDFOP : public BaseSDFOP 
{ 
public: 
    Capsule_SDFOP(); 
    ~Capsule_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
}; 
 
#endif // CAPSULE_SDFOP_H 

