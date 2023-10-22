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

    bool isReady();


    uint16_t computeMove(uint32_t simulations);

    Node* getCurrentNode();
    Node* getParentNode();

    bool isTerminal();

    void clean();

private:
    std::vector<Node*> deletion_queue;
    Model* neural_net;
    Node* root_node;
    Node* current_node;
};