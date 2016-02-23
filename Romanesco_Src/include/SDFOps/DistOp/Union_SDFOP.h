
#ifndef UNION_SDFOP_H 
#define UNION_SDFOP_H 
 
#include "DistOpInterface_SDFOP.h"
 
class Union_SDFOP : public DistOpInterface_SDFOP
{ 
public: 
    Union_SDFOP(); 
    ~Union_SDFOP();

    virtual std::string getDefaultArg(unsigned int index);
}; 
 
#endif // UNION_SDFOP_H 

