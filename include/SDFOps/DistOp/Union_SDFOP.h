
#ifndef UNION_SDFOP_H 
#define UNION_SDFOP_H 
 
#include "DistOpInterface_SDFOP.h"
 
class Union_SDFOP : public BaseSDFOP
{ 
public: 
    Union_SDFOP(); 
    ~Union_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // UNION_SDFOP_H 

