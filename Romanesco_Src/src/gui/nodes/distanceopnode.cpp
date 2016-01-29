#include "include/gui/nodes/distanceopnode.h"
#include "qneport.h"

DistanceOpNode::DistanceOpNode(const QString &_name, QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent)
{
    addPort(_name, 0, QNEPort::NamePort);
    addPort("Distance", 0, QNEPort::TypePort);

    addInputPort("A");
    addInputPort("B");

    addOutputPort("Out");
}

DistanceOpNode::~DistanceOpNode()
{
}
