#pragma once

#include "Config.h"
#include "Model.h"
#include "Node.h"
#include "State.h"

class Tree
{
public:
    Tree();
    ~Tree();

    // These functions can/will require a NN computation, those will be stored in network queue
    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(uint16_t index);
    Node* simulationStep();
    // <-------------------->

    // Network queue managment
    std::vector<Node*> getNetworkQueue();
    bool clearNetworkQueue();
    
    Node* getCurrentNode();
    Node* getParentNode();

    bool isTerminal();
    void clean();

private:
    std::vector<Node*> deletion_queue;
    std::vector<Node*> network_queue;
    Node* root_node;
    Node* current_node;
};