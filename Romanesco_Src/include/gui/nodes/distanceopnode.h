#ifndef DISTANCEOPNODE_H
#define DISTANCEOPNODE_H

#include "qneblock.h"

class DistanceOpNode : public QNEBlock
{
public:
    DistanceOpNode(const QString &_name, QGraphicsScene* _scene, QGraphicsItem *parent = 0);
    ~DistanceOpNode();

};

#endif // DISTANCEOPNODE_H
