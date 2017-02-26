#ifndef DISTANCEOPNODE_H
#define DISTANCEOPNODE_H

#include "nodegraph/qneblock.h"

#include "Base_SDFOP.h"

#include "Blend_SDFOP.h"
#include "Displace_SDFOP.h"

#include "Intersection_SDFOP.h"
#include "Subtraction_SDFOP.h"
#include "Union_SDFOP.h"

#include "Repetition_SDFOP.h"
#include "Scale_SDFOP.h"
#include "Transform_SDFOP.h"

#include "Bend_SDFOP.h"
#include "Twist_SDFOP.h"

#include "Box_SDFOP.h"
#include "Capsule_SDFOP.h"
#include "Cone_SDFOP.h"
#include "Cylinder_SDFOP.h"
#include "Mandelbulb_SDFOP.h"
#include "Menger_SDFOP.h"
#include "Sphere_SDFOP.h"
#include "Torus_SDFOP.h"

class DistanceOpNode : public QNEBlock
{
public:
    DistanceOpNode(const QString &_name, QGraphicsScene* _scene, QGraphicsItem *parent = 0);
    DistanceOpNode(BaseSDFOP* _op, QGraphicsScene* _scene, QGraphicsItem *parent = 0);
    ~DistanceOpNode();

    BaseSDFOP* getSDFOP() { return m_op; }

private:
    BaseSDFOP* m_op;
};

#endif // DISTANCEOPNODE_H
