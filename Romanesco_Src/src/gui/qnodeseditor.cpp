/* Copyright (c) 2012, STANISLAW ADASZEWSKI
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of STANISLAW ADASZEWSKI nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL STANISLAW ADASZEWSKI BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#include "qnodeseditor.h"

#include <stdexcept>
#include <iostream>

#include <QGraphicsScene>
#include <QEvent>
#include <QGraphicsSceneMouseEvent>

#include "qneport.h"
#include "qneconnection.h"
#include "qneblock.h"
#include "nodes/terminatenode.h"

#include <QDebug>

QNodesEditor::QNodesEditor(QObject *parent) :
    QObject(parent)
{
	conn = 0;
}

void QNodesEditor::install(QGraphicsScene *s)
{
	s->installEventFilter(this);
	scene = s;

    endBlock = new TerminateNode(scene, 0);
}

QGraphicsItem* QNodesEditor::itemAt(const QPointF &pos)
{
	QList<QGraphicsItem*> items = scene->items(QRectF(pos - QPointF(1,1), QSize(3,3)));

	foreach(QGraphicsItem *item, items)
		if (item->type() > QGraphicsItem::UserType)
			return item;

	return 0;
}

#include <QKeyEvent>
#include "gui/nodes/distanceopnode.h"

bool QNodesEditor::eventFilter(QObject *o, QEvent *e)
{
	QGraphicsSceneMouseEvent *me = (QGraphicsSceneMouseEvent*) e;

	switch ((int) e->type())
	{
        case QEvent::KeyPress:
            {
                QKeyEvent *k = static_cast<QKeyEvent *>(e);
                switch(k->key())
                {
                    case Qt::Key_P:
                        {
                            std::vector<Backpass> backpasses;

                            this->getItems(endBlock, backpasses);

                            auto nodes = this->getNodeList();

                            int i = 0;
                            qDebug() << "Node Table";
                            for(auto node: nodes)
                            {
                                qDebug() << i++ << " : " << node->displayName();
                            }
                            qDebug() << "\n";

                            std::reverse(backpasses.begin(), backpasses.end());
                            for(auto pass: backpasses)
                            {
                                std::string indent = "";// std::string(nodeCtr, '\t').c_str();

                                auto currentNode = std::find(nodes.begin(), nodes.end(), pass.currentNodePtr);
                                if ( currentNode == nodes.end())
                                {
                                    continue;
                                }

                                qDebug().nospace() << indent.c_str() << "Node Name: " << qPrintable(pass.currentNodePtr->displayName()) << " (" << std::distance(nodes.begin(), currentNode) << ")";
                                qDebug().nospace() << indent.c_str() << "Depth: " << pass.nodeCtr;

//                                int i = 0;
                                for(auto node: pass.inputNodes)
                                {
                                    auto nodeTmp = std::find(nodes.begin(), nodes.end(), node.second );
                                    if ( nodeTmp != nodes.end())
                                    {
                                        //" Input " << i << ": " << pStartConnection->block()->displayName() << ":" << pStartConnection->portName();
//                                        qDebug().nospace() << indent.c_str() << "\t[" << i << "]:" << qPrintable(node->displayName());
                                        qDebug().nospace() << indent.c_str() << "\t Input[" << node.first << "]:" << std::distance(nodes.begin(), nodeTmp) << " (" << (*nodeTmp)->displayName() << ")";
                                    }
                                }
                            }

                            break;
                        }
                    case Qt::Key_C:
                        {
                            DistanceOpNode *c = new DistanceOpNode("Union", scene, 0);
                            break;
                        }
                }
                break;
            }
        case QEvent::GraphicsSceneMousePress:
            {
                switch ((int) me->button())
                {
                    case Qt::LeftButton:
                        {
                            QGraphicsItem *item = itemAt(me->scenePos());
                            if (item && item->type() == QNEPort::Type)
                            {
                                conn = new QNEConnection(0);
                                scene->addItem(conn);
                                conn->setPort1((QNEPort*) item);
                                conn->setPos1(item->scenePos());
                                conn->setPos2(me->scenePos());
                                conn->updatePath();

                                return true;
                            } else if (item && item->type() == QNEBlock::Type)
                            {
                                /* if (selBlock)
                    selBlock->setSelected(); */
                                // selBlock = (QNEBlock*) item;
                            }
                            break;
                        }
                    case Qt::RightButton:
                        {
                            QGraphicsItem *item = itemAt(me->scenePos());
                            if (item && (item->type() == QNEConnection::Type || item->type() == QNEBlock::Type))
                                delete item;
                            // if (selBlock == (QNEBlock*) item)
                            // selBlock = 0;

                            break;
                        }
                }
            }
        case QEvent::GraphicsSceneMouseMove:
            {
                if (conn)
                {
                    conn->setPos2(me->scenePos());
                    conn->updatePath();
                    return true;
                }
                break;
            }
        case QEvent::GraphicsSceneMouseRelease:
            {
                if (conn && me->button() == Qt::LeftButton)
                {
                    QGraphicsItem *item = itemAt(me->scenePos());
                    if (item && item->type() == QNEPort::Type)
                    {
                        QNEPort *port1 = conn->port1();
                        QNEPort *port2 = (QNEPort*) item;

                        if (port1->block() != port2->block() && port1->isOutput() != port2->isOutput() && !port1->isConnected(port2))
                        {
                            conn->setPos2(port2->scenePos());
                            conn->setPort2(port2);
                            conn->updatePath();
                            conn = 0;
                            return true;
                        }
                    }

                    delete conn;
                    conn = 0;
                    return true;
                }
                break;
            }
    }
    return QObject::eventFilter(o, e);
}

static const std::string example = R"DELIM(
                                Eval "Out" Input 0: "Union_0":"Out"
                                    Eval "Union_0" Input 0: "Union_1":"Out"
                                        Eval "Union_1" Input 0: "Union_4":"Out"
                                        Eval "Union_1" Input 0: "Union_3":"Out"
                                    Eval "Union_0" Input 0: "Union_2":"Out"
                                        Eval "Union_2" Input 0: "Union_5":"Out"
                                )DELIM";



NodeList QNodesEditor::getNodeList()
{
    NodeList nodes;

    foreach(QGraphicsItem *item, scene->items())
    {
        QNEBlock* node = qgraphicsitem_cast<QNEBlock*>(item);
        if(node)
        {
            nodes.emplace_back(node);
        }
    }

    return nodes;
}

void QNodesEditor::getItems(QNEBlock* _node, std::vector<Backpass>& backpasses, int _depth)
{
    std::string indent = std::string(_depth, '\t').c_str();
    //qDebug().nospace() << indent.c_str() << "Eval " << _node->displayName();

    NodeList nodes = getNodeList();
//    for(auto node: nodes)
//    {
//        qDebug() << node->displayName();
//    }

    Backpass pass;
    pass.nodeCtr = _depth;
    pass.currentNodePtr = _node;

    auto ports = _node->inputPorts().toStdVector();

    //std::reverse(ports.begin(), ports.end());

    int i = 0;
    // Iterate over all input ports on _node
    for(QNEPort* port : ports)
    {
        //@todo BUG : Left->Right connections work okay, Right->Left erroneously attach the node to it's own input (infinite loop)
        std::vector<QNEConnection*> connections = port->connections().toStdVector();

        //std::reverse(connections.begin(), connections.end());

        // Iterate over all connections to port (probably only one)
        //for(int i = 0; i < connections.size(); i++)
        if(connections.size() > 0)
        {
            const QNEConnection* connection = connections[0];

            const QNEPort* pStartConnection = connection->port1();
            const QNEPort* pEndConnection = connection->port2();

            pass.inputNodes.insert( std::make_pair(i, pStartConnection->block()) );

            if(_depth > 1024)
            {
                throw std::overflow_error("Maximum node depth exceeded");
            }

            getItems(pStartConnection->block(), backpasses, _depth + 1);
        }

        i++;
    }

    backpasses.push_back(pass);

//    foreach(QGraphicsItem *item, scene->items())
//    {
//        QNEBlock* node = qgraphicsitem_cast<QNEBlock*>(item);
//        if(node)
//        {
//            qDebug() << node->displayName();
//            for(auto port : node->ports())
//            {
//                QVector<QNEConnection*>& connections = port->connections();
//                for(QNEConnection* connection : connections)
//                {
//                    QNEPort* p1 = connection->port1();
//                    QNEPort* p2 = connection->port2();

//                    qDebug() << "\t" << p1->block()->displayName() << "->" << p2->block()->displayName();
//                }
//            }
//        }
//    }

//    foreach(QGraphicsItem *item, scene->items())
//    {
//        QNEBlock* node = qgraphicsitem_cast<QNEBlock*>(item);
//        if(node)
//        {
//            qDebug() << node->displayName();
//            for(auto port : node->ports())
//            {
//                QVector<QNEConnection*>& connections = port->connections();
//                for(QNEConnection* connection : connections)
//                {
//                    QNEPort* p1 = connection->port1();
//                    QNEPort* p2 = connection->port2();

//                    qDebug() << "\t" << p1->block()->displayName() << "->" << p2->block()->displayName();
//                }
//            }
//        }
//    }

//    foreach(QGraphicsItem *item, scene->items())
//    {
//        if (item->type() == QNEConnection::Type)
//        {
//            qDebug() << item->type() - QGraphicsItem::UserType;
////                ((QNEConnection*) item)->save(ds);
//        }
//    }
}


void QNodesEditor::save(QDataStream &ds)
{
	foreach(QGraphicsItem *item, scene->items())
		if (item->type() == QNEBlock::Type)
		{
			ds << item->type();
			((QNEBlock*) item)->save(ds);
		}

	foreach(QGraphicsItem *item, scene->items())
		if (item->type() == QNEConnection::Type)
		{
			ds << item->type();
			((QNEConnection*) item)->save(ds);
		}
}

void QNodesEditor::load(QDataStream &ds)
{
	scene->clear();

	QMap<quint64, QNEPort*> portMap;

	while (!ds.atEnd())
	{
		int type;
		ds >> type;
		if (type == QNEBlock::Type)
		{
            QNEBlock *block = new QNEBlock(0);
            scene->addItem(block);
			block->load(ds, portMap);
		} else if (type == QNEConnection::Type)
		{
            QNEConnection *conn = new QNEConnection(0);
            scene->addItem(conn);
			conn->load(ds, portMap);
		}
	}
}
