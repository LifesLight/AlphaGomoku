#pragma once

#include "Utilities.h"
#include "Node.h"
#include "Model.h"
#include "State.h"
#include "Tree.h"

/*
For batching, we take one batch out of every parellel simulation step
When calling make move there is a chance that we need to call the model,
so we take all those into a batch if needed;

For each environment we store 
*/


// Environment is a wrapper around 2 Trees, most functions act simmelar
class Environment
{
public:
    Environment();
    ~Environment();

    // These functions can/will require a NN computation, those will be stored in trees netqueues
    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(uint16_t index);
    bool makeBestMove();
    bool makeRandomMove();
    Node* policy();
    // <-------------------->

    // These are wrappers around tree, it just translates it to the env
    // This network queue has a tuple to with network it needs to be run on
    std::vector<std::tuple<Node*, bool>> getNetworkQueue();
    bool clearNetworkQueue();

    // Display state
    std::string toString();
    std::string toString(uint8_t depth);

    bool isTerminal();
    bool getNextColor();

    // Default current
    Node* getCurrentNode();
    Node* getOpposingNode();

    void freeMemory();

private:
    Tree* trees[2] = {nullptr, nullptr};
    bool next_color;
};