#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


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
    Environment(bool dual_tree);
    ~Environment();

    // These functions can/will require a NN computation, those will be stored in trees netqueues
    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(index_t index);
    bool makeBestMove();
    Node* policy();
    // <-------------------->

    std::vector<Node*> getRootNodes();

    // These are wrappers around tree, it just translates it to the env
    // This network queue has a tuple to with network it needs to be run on
    std::vector<std::tuple<Node*, bool>> getNetworkQueue();
    bool clearNetworkQueue();
    void forceClearNetworkQueue();
    // Black is 0, White is 1, Draw is 2
    StateResult getResult();

    // Get all possible actions from node
    std::vector<index_t> getUntriedActions();

    // Display state
    std::string toString();
    std::string toString(uint8_t depth);

    bool isTerminal();
    StateColor getNextColor();

    // Collapses all trees
    void collapseEnvironment();

    // Default current
    Node* getCurrentNode();
    // Returns null on single tree env
    Node* getOpposingNode();

    // Just swaps how the model index in getNetworkQueue is set
    void swapModels();
    bool areModelsSwapped();

    void freeMemory();

private:
    Tree* trees[2];
    bool next_color;
    bool swapped_models;
};