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
#include <sstream>

#include <QGraphicsScene>
#include <QEvent>
#include <QGraphicsSceneMouseEvent>

#include "qneport.h"
#include "qneconnection.h"
#include "qneblock.h"
#include "nodes/terminatenode.h"

#include <QDebug>

QNodeGraph::QNodeGraph(QObject *parent) :
    QObject(parent)
{
	conn = 0;
}

void QNodeGraph::install(QGraphicsScene *s)
{
	s->installEventFilter(this);
	scene = s;

    endBlock = new TerminateNode(scene, 0);
}

QGraphicsItem* QNodeGraph::itemAt(const QPointF &pos)
{
	QList<QGraphicsItem*> items = scene->items(QRectF(pos - QPointF(1,1), QSize(3,3)));

	foreach(QGraphicsItem *item, items)
		if (item->type() > QGraphicsItem::UserType)
			return item;

	return 0;
}

#include <QKeyEvent>




bool QNodeGraph::eventFilter(QObject *o, QEvent *e)
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
                            parseGraph();

                            break;
                        }
                    case Qt::Key_C:
                        {
                            BaseSDFOP* op = new Union_SDFOP();
                            DistanceOpNode *c = new DistanceOpNode(op, scene, 0);

                            emit graphChanged();

                            break;
                        }
                    case Qt::Key_V:
                        {
                            BaseSDFOP* op = new Sphere_SDFOP(1.0f);
                            DistanceOpNode *c = new DistanceOpNode(op, scene, 0);

                            emit graphChanged();

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

//                                emit graphChanged();

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
                            {
                                delete item;
                                emit graphChanged();
                            }
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
        emit graphChanged();
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

                            emit graphChanged();

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



std::vector<DistanceOpNode *> QNodeGraph::getNodeList()
{
    std::vector<DistanceOpNode*> nodes;

    foreach(QGraphicsItem *item, scene->items())
    {
        DistanceOpNode* node = qgraphicsitem_cast<DistanceOpNode*>(item);
        if(node)
        {
            nodes.emplace_back(node);
        }
    }

    return nodes;
}

std::string QNodeGraph::parseGraph()
{
    static const std::string hit_globals = R"(
// Input Globals
//__device__ float3 P;
//__device__ float T;
//__device__ float MaxIterations;
)";

    std::string hit_src = R"(
extern "C" {
__device__ float distancehit_hook(float3 x, float _t, float _max_iterations)
{
            // Initialise globals
//            P = x;
//            T = _t;
//            MaxIterations = _max_iterations;
)";

    std::ostringstream resultStream;

    /* Parse node graph backwards first */
    std::vector<BackwardPass> backpasses;

    // Recursively parse backwards from the endblock and fill up the backpass array
    this->backwardsParse(endBlock, backpasses);

    std::vector<DistanceOpNode*> nodes = this->getNodeList();

    // We need to define all the functions we're using just once, so store them
    // in the map
    std::map<std::string, bool> functionDefinitionMap;

    // Generate all the includes
    for(const auto& header: BaseSDFOP::m_headers)
    {
//        qDebug( "#include \"%s\"", header.c_str() );
        resultStream << "#include \"" << header << "\"\n";
    }

    // Generate the global variables
    //qDebug() << hit_globals.c_str();
    //resultStream << hit_globals << "\n";

    int i = 0;
    for(auto node: nodes)
    {
        //qDebug() << i++ << " : " << node->displayName();

        BaseSDFOP* nodeSDFOP = node->getSDFOP();

        std::string args = "";

        if(nodeSDFOP)
        {
            const std::string nodeFunctionName = nodeSDFOP->getFunctionName();
            bool isFunctionDeclared = functionDefinitionMap.count(nodeFunctionName) > 0;

            // Only write out the function declaration if we haven't already
            if(!isFunctionDeclared)
            {
                // Join the argument string
                std::stringstream ss;
                for(unsigned int i = 0; i < nodeSDFOP->argumentSize(); i++)
                {
                    if(i != 0) {
                      ss << ",";
                    }

                    Argument arg = nodeSDFOP->getArgument(i);
                    ss << ReturnLookup[ (int)arg.type ] << " " << arg.name;
                }

                args = ss.str();

                resultStream << "\n";
                resultStream << nodeSDFOP->getTypeString() << " " << nodeSDFOP->getFunctionName() << "(" << args << ")\n";
                resultStream << "{\n" << nodeSDFOP->getSource() << "\n}\n";

//                qDebug("%s %s(%s)", nodeSDFOP->getTypeString().c_str(), nodeSDFOP->getFunctionName().c_str(), args.c_str());
//                qDebug("{\n%s\n}", nodeSDFOP->getSource().c_str());

                // Mark the function as being declared (not the best data structure to use for this I think but it works)
                functionDefinitionMap.insert( std::pair<std::string, bool>(nodeSDFOP->getFunctionName(), true ) );
            }
        }
    }
    //qDebug() << "\n";
    resultStream << "\n";

    std::vector<ForwardPass> forward;
    //std::reverse(backpasses.begin(), backpasses.end());
    i = 0;
    for(auto pass: backpasses)
    {
        ForwardPass tmp;
        tmp.index = i++;

        std::string indent = "";// std::string(nodeCtr, '\t').c_str();

        auto currentNode = std::find(nodes.begin(), nodes.end(), pass.currentNodePtr);
        if ( currentNode == nodes.end())
        {
            continue;
        }

        tmp.currentNodePtr = *currentNode;

        for(auto node: pass.inputNodes)
        {
            auto nodeTmp = std::find(nodes.begin(), nodes.end(), node.second );
            if ( nodeTmp != nodes.end())
            {
                //" Input " << i << ": " << pStartConnection->block()->displayName() << ":" << pStartConnection->portName();
//                                        qDebug().nospace() << indent.c_str() << "\t[" << i << "]:" << qPrintable(node->displayName());
                //qDebug().nospace() << indent.c_str() << "\t Input[" << node.first << "]:" << std::distance(nodes.begin(), nodeTmp) << " (" << (*nodeTmp)->displayName() << ")";
                tmp.inputs.insert( std::make_pair(node.first, nodes[std::distance(nodes.begin(), nodeTmp)] ) );
            }
        }

        forward.push_back(tmp);
    }

    // Lookup the code variable based on the node's name (which is unique for now)
    std::unordered_map<std::string, std::string> variableMap;

    //qDebug() << hit_src.c_str();
    resultStream << hit_src;

    //std::reverse(forward.begin(), forward.end());
    for(ForwardPass pass: forward)
    {
        std::string nodeName = qPrintable( pass.currentNodePtr->displayName() );

        // Generate a unique variable name to assign, right now it just uses alphabetical ascii names
        std::string varName(1,  (char)pass.index + 97);

        variableMap.insert( std::pair<std::string, std::string>(nodeName, varName) );

        BaseSDFOP* nodeSDFOP = pass.currentNodePtr->getSDFOP();
        unsigned int totalInputs;
        // Determine if we're the 'end' node or not by checking if the SDFOP is null
        ////@todo Make this more robust, but it works for now
        if(nodeSDFOP)
        {
            // Take into account default arguments if we are a valid SDFOP
            totalInputs = nodeSDFOP->argumentSize();
        } else {
            // Otherwise use the actual inputs (for the return node this should only be 1, and all other nodes ~should~ be SDFOPS)
            totalInputs = pass.inputs.size();
        }

        std::string args;

        {
            std::stringstream ss;
            for(unsigned int i = 0; i < totalInputs; i++)
            {
                if(i != 0) {
                    ss << ",";
                }

                // Inputs are mapped from int->node, as it is possible to have no connection
                // and we need to know which inputs are 'empty' so we can fill in the default argument
                auto inputMap = pass.inputs.find( i );

                // Get connected input value if there is one
                if(inputMap != pass.inputs.end())
                {
                    DistanceOpNode* node = inputMap->second;
                    std::string currentNodeName = qPrintable( node->displayName() );
                    ss << variableMap[currentNodeName];
                }
                else // otherwise get the default value for this argument
                {
                    Argument arg = nodeSDFOP->getArgument(i);
                    ss << arg.defaultValue;
                }
            }

            args = ss.str();
        }

        // check if we are an SDFOP or a terminate node
        if( nodeSDFOP )
        {
            // Call the SDF function (with the appropriate arguments) and store the result in it's variable
            BaseSDFOP* arg = pass.currentNodePtr->getSDFOP();
            const std::string variableType = arg->getTypeString();
            const std::string variableName = varName;
            const std::string functionName = arg->getFunctionName();

//            qDebug("   %s %s = %s(%s);", variableType.c_str(), variableName.c_str(), functionName.c_str(), args.c_str() );

            resultStream << "\t" << variableType << " " << variableName << " = " << functionName << "(" << args << ");\n";
        }
        else // Assume we're a terminate node, so return the final result we calculated
        {
            if(totalInputs == 0)
            {
                // Special case to return nothing if no nodes are connected up yet
                args = "0";
            }

//            qDebug("   return %s;", args.c_str() );
            resultStream << "\t" << "return " << args << ";\n";
        }
    }

    //qDebug() << "}";
    resultStream << "}\n}\n";

    return resultStream.str();
}

void QNodeGraph::backwardsParse(QNEBlock* _node, std::vector<BackwardPass>& backpasses, int _depth)
{
    std::string indent = std::string(_depth, '\t').c_str();
    //qDebug().nospace() << indent.c_str() << "Eval " << _node->displayName();

    BackwardPass pass;
    pass.nodeCtr = _depth;
    pass.currentNodePtr = _node;

    std::vector<QNEPort*> ports = _node->inputPorts().toStdVector();

    int i = 0;
    // Iterate over all input ports on _node and fill up the inputNodes of the current pass
    for(QNEPort* port : ports)
    {
        //@todo BUG : Left->Right connections work okay, Right->Left erroneously attach the node to it's own input (infinite loop)
        std::vector<QNEConnection*> connections = port->connections().toStdVector();

        if(connections.size() > 0)
        {
            // We only care about the first connected node
            const QNEConnection* connection = connections[0];

            const QNEPort* pStartConnection = connection->port1();
//            const QNEPort* pEndConnection = connection->port2();

            pass.inputNodes.insert( std::make_pair(i, pStartConnection->block()) );

            if(_depth > 1024)
            {
                throw std::overflow_error("Maximum node depth exceeded");
            }

            // Recursively check the connected node with the same process
            backwardsParse(pStartConnection->block(), backpasses, _depth + 1);
        }

        i++;
    }

    backpasses.push_back(pass);
}


void QNodeGraph::save(QDataStream &ds)
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

void QNodeGraph::load(QDataStream &ds)
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
