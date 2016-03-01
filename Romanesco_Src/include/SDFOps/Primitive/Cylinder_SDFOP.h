
#ifndef CYLINDER_SDFOP_H 
#define CYLINDER_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Cylinder_SDFOP : public BaseSDFOP 
{ 
public: 
    Cylinder_SDFOP(); 
    ~Cylinder_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
}; 
 
#endif // CYLINDER_SDFOP_H 

