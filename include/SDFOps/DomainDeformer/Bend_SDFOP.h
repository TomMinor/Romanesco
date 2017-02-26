
#ifndef BEND_SDFOP_H 
#define BEND_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Bend_SDFOP : public BaseSDFOP 
{ 
public: 
    Bend_SDFOP(); 
    ~Bend_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // BEND_SDFOP_H 

