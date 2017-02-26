
#ifndef CONE_SDFOP_H 
#define CONE_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Cone_SDFOP : public BaseSDFOP 
{ 
public: 
    Cone_SDFOP(); 
    ~Cone_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // CONE_SDFOP_H 

