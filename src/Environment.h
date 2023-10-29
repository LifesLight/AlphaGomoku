#pragma once

#include "Utilities.h"
#include "Node.h"
#include "Model.h"
#include "State.h"
#include "Tree.h"

/*
Environment stores 2 Trees and syncs them so that each Model / Agent has his own Tree 
(Relevant for not needing to store different values for "same" node for 2 different models in each Tree)

Also has a queue for deleting obsolete nodes, this might not be neccessary.
*/

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
    // Black is 0, White is 1, Draw is 2
    uint8_t getResult();

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