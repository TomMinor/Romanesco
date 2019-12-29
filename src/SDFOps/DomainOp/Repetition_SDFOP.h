
#ifndef REPETITION_SDFOP_H 
#define REPETITION_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Repetition_SDFOP : public BaseSDFOP 
{ 
public: 
    Repetition_SDFOP(); 
    ~Repetition_SDFOP(); 

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;
}; 
 
#endif // REPETITION_SDFOP_H 

