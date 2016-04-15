#include "include/gui/nodes/terminatenode.h"
#include "nodegraph/qneport.h"

TerminateNode::TerminateNode(QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent)
{
    addPort("Out", 0, QNEPort::NamePort);
    addPort("Result", 0, QNEPort::TypePort);

    addInputPort("Distance");
}

TerminateNode::~TerminateNode()
{

}
