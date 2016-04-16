#ifndef NODE_PARSER__
#define NODE_PARSER__

#include "nodegraph/qneport.h"
#include "nodegraph/qneconnection.h"
#include "nodegraph/qneblock.h"

class NodeParser
{
public:
    NodeParser();

protected:
    virtual void getNextSymbol();

private:


};

#endif
