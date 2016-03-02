
#ifndef BOX_SDFOP_H 
#define BOX_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Box_SDFOP : public BaseSDFOP 
{ 
public: 
    Box_SDFOP(); 
    ~Box_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // BOX_SDFOP_H 

