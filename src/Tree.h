#pragma once

#include "Config.h"
#include "Model.h"
#include "Node.h"
#include "State.h"
#include "Gamestate.h"

class Tree
{
public:
    Tree(Model* neural_network);
    ~Tree();

    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(uint16_t index);
    void simulationStep();
    uint16_t bestMove();

    Node* getCurrentNode();
    Node* getParentNode();

    bool isReady();
    std::vector<Node*> getNetworkQueue();
    void clearNetworkQueue();

    bool isTerminal();
    void clean();

private:
    std::vector<Node*> deletion_queue;
    std::vector<Node*> network_queue;
    Model* neural_net;
    Node* root_node;
    Node* current_node;
};