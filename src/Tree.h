#pragma once

#include "Config.h"
#include "Model.h"
#include "Node.h"
#include "State.h"

/*
A Tree is a domain for a singular Node tree, its main purpose is to make environment simpler.

Tree also automatically accumilates a network queue which is a list of nodes still requiring model data.
*/

class Tree
{
public:
    Tree();
    ~Tree();

    // These functions can/will require a NN computation, those will be stored in network queue
    void makeMove(uint8_t x, uint8_t y);
    void makeMove(index_t index);
    Node* policy();
    // <-------------------->

    Node* getRootNode();

    // Network queue managment
    std::list<Node*> getNetworkQueue();
    bool clearNetworkQueue();
    void forceClearNetworkQueue();

    Node* getCurrentNode();
    Node* getParentNode();

    int getNodeCount();
    std::vector<Node*> getAllNodes();

    bool isTerminal();
    void clean();

private:
    void updateCurrentNode(index_t action);

    std::list<Node*> deletion_queue;
    std::list<Node*> network_queue;
    Node* root_node;
    Node* current_node;
    int node_count;
};