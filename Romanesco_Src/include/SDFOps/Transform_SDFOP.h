#ifndef TRANSFORM_SDFOP_H 
#define TRANSFORM_SDFOP_H 
 
#include <glm/vec3.hpp>

#include "Base_SDFOP.h"
 
class Transform_SDFOP : public BaseSDFOP 
{ 
public: 
    Transform_SDFOP(const glm::vec3 &_m);
    ~Transform_SDFOP();

    virtual std::string getSource();

private:
    glm::vec3 m_transform;
}; 
 
#endif // TRANSFORM_SDFOP_H 

