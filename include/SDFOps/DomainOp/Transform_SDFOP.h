#ifndef TRANSFORM_SDFOP_H 
#define TRANSFORM_SDFOP_H 
 
#include <glm/vec3.hpp>

#include "Base_SDFOP.h"
 
class Transform_SDFOP : public BaseSDFOP 
{ 
public: 
    Transform_SDFOP(const glm::vec3 &_m);
    ~Transform_SDFOP();

    virtual std::string getFunctionName() override;
    virtual std::string getSource() override;
    virtual Argument getArgument(unsigned int index) override;
    virtual unsigned int argumentSize() override;

private:
    glm::vec3 m_transform;
}; 
 
#endif // TRANSFORM_SDFOP_H 

