
#ifndef INTERSECTION_SDFOP_H 
#define INTERSECTION_SDFOP_H 
 
#include "DistOpInterface_SDFOP.h"
 
class Intersection_SDFOP : public DistOpInterface_SDFOP
{ 
public: 
    Intersection_SDFOP();
    ~Intersection_SDFOP(); 

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // INTERSECTION_SDFOP_H 

