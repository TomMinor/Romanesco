#ifndef TERMINATENODE_H
#define TERMINATENODE_H

#include "qneblock.h"


class TerminateNode : public QNEBlock
{
public:
    TerminateNode(QGraphicsScene* _scene, QGraphicsItem *parent = 0);
    ~TerminateNode();
};

#endif // TERMINATENODE_H
