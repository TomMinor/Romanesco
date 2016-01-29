#include "include/gui/nodes/distanceopnode.h"
#include "qneport.h"

DistanceOpNode::DistanceOpNode(QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent)
{
    addPort("Distance", 0, QNEPort::NamePort);
    addPort("DistanceOp", 0, QNEPort::TypePort);

    addInputPort("A");
    addInputPort("B");
    addInputPort("B");
    addInputPort("B");


    addOutputPort("Out");
    addOutputPort("Out");
    addOutputPort("Out");
    addOutputPort("Out");
}

DistanceOpNode::~DistanceOpNode()
{
}
