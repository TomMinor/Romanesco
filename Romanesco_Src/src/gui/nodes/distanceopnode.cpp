#include "include/gui/nodes/distanceopnode.h"
#include "qneport.h"

static int ctr = 0;

DistanceOpNode::DistanceOpNode(BaseSDFOP* _op, QGraphicsScene* _scene, QGraphicsItem *parent)
    : QNEBlock(_scene, parent), m_op(_op)
{
    QString name = QString("%1_%2").arg( m_op->getFunctionName().c_str() ).arg(ctr++);

    addPort(name, 0, QNEPort::NamePort);
    addPort("Distance", 0, QNEPort::TypePort);

    for(unsigned int i = 0; i < m_op->argumentSize(); i++)
    {
        try
        {
            Argument arg = _op->getArgument(i);
            addInputPort( arg.name.c_str() );
        }
        catch ( const std::out_of_range e )
        {
            qWarning("Invalid argument accessed at index %d, ignoring", i);
            continue;
        }

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
