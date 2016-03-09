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

#ifndef QNODESEDITOR_H
#define QNODESEDITOR_H

#include <QObject>
#include <vector>
#include <map>
#include <unordered_map>

class QGraphicsScene;
class QNEConnection;
class QGraphicsItem;
class QPointF;
class QNEBlock;
class DistanceOpNode;


#include "gui/nodes/distanceopnode.h"

namespace {

///
/// \brief The Backpass struct stores node data in the context of parsing the nodegraph backwards
///
struct BackwardPass
{
    ///
    /// \brief nodeCtr
    ///
    int nodeCtr = 0;

    ///
    /// \brief currentNodePtr
    ///
    QNEBlock* currentNodePtr = nullptr;

    ///
    /// \brief inputNodes
    ///
    std::map<unsigned int, QNEBlock* > inputNodes;
};

///
/// \brief The ForwardPass struct
///
struct ForwardPass
{
    ///
    /// \brief index A unique index, used to map variable names to node return values
    ///
    unsigned int index;

    ///
    /// \brief node
    ///
    DistanceOpNode* node;

    ///
    /// \brief inputs Map the index of an input to the connected node,
    /// it is possible to have 'empty' inputs, missing inputs will look up
    /// the default value for the respective argument in the SDFOP of the node
    ///
    std::map<unsigned int, DistanceOpNode*> inputs;

    ///
    /// \brief expectedInputs
    ///
    unsigned int expectedInputs;
};

}

class QNodeGraph : public QObject
{
	Q_OBJECT
public:
    explicit QNodeGraph(QObject *parent = 0);

	void install(QGraphicsScene *scene);

	bool eventFilter(QObject *, QEvent *);


	void save(QDataStream &ds);
	void load(QDataStream &ds);


    std::string parseGraph();

    void backwardsParse(QNEBlock *_node, std::vector<BackwardPass> &backpasses, int _depth = 0);
private:
	QGraphicsItem *itemAt(const QPointF&);

    std::vector<DistanceOpNode*> getNodeList();

private:
	QGraphicsScene *scene;
	QNEConnection *conn;

    QNEBlock *endBlock;
};

#endif // QNODESEDITOR_H
