#include "include/gui/nodes/distanceopnode.h"
#include "qneport.h"

static int ctr = 0;

DistanceOpNode::DistanceOpNode(const QString &_name, QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent)
{
    QString name = QString("%1_%2").arg(_name).arg(ctr++);
    addPort(name, 0, QNEPort::NamePort);
    addPort("Distance", 0, QNEPort::TypePort);

    addInputPort("A");
    addInputPort("B");

    addOutputPort("Out");
}

DistanceOpNode::~DistanceOpNode()
{
}
