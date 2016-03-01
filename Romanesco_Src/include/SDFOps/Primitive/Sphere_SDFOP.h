
#ifndef SPHERE_SDFOP_H 
#define SPHERE_SDFOP_H 
 
#include "Base_SDFOP.h" 
 
class Sphere_SDFOP : public BaseSDFOP 
{ 
public: 
    Sphere_SDFOP(float _radius);
    ~Sphere_SDFOP(); 

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;

private:
    float m_radius;
}; 
 
#endif // SPHERE_SDFOP_H 

