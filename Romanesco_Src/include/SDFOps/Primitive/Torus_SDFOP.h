
#ifndef TORUS_SDFOP_H 
#define TORUS_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Torus_SDFOP : public BaseSDFOP 
{ 
public: 
    Torus_SDFOP(); 
    ~Torus_SDFOP(); 

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // TORUS_SDFOP_H 

