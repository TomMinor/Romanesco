#include "include/gui/nodes/distanceopnode.h"
#include "qneport.h"

static int ctr = 0;

DistanceOpNode::DistanceOpNode(BaseSDFOP* _op, QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent), m_op(_op)
{
    QString name = QString("%1_%2").arg( m_op->getFunctionName().c_str() ).arg(ctr++);

    addPort(name, 0, QNEPort::NamePort);
    addPort("Distance", 0, QNEPort::TypePort);

    auto a =  m_op->argumentSize();

    for(unsigned int i = 0; i <= m_op->argumentSize(); i++)
    {
        Argument arg = _op->getArgument(i);
        addInputPort( arg.name.c_str() );
    }

//    addInputPort("A");
//    addInputPort("B");

    addOutputPort("Out");
}

DistanceOpNode::DistanceOpNode(const QString &_name, QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent), m_op(nullptr)
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
