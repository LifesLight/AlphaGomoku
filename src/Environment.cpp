/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Environment.h"

Environment::Environment(bool dual_tree)
    : next_color(false), swapped_models(false)
{ 
    trees[1] = nullptr;

    // We only init one tree if not in dual tree mode
    for (int i = 0; i < dual_tree + 1; i++)
        trees[i] = new Tree();
}

Environment::~Environment()
{
    // Delete trees
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
            delete trees[i];
}

bool Environment::makeMove(index_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}

bool Environment::makeMove(uint8_t x, uint8_t y)
{
    bool success = true;
    // Update all existing trees
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
            if (!trees[i]->makeMove(x, y))
                success = false;

    if (success)
        next_color = !next_color;

    return success;
}

std::vector<Node*> Environment::getRootNodes()
{
    std::vector<Node*> nodes;
    for (int i = 0; i < 2; i++)
        if (trees[i] != nullptr)
            nodes.push_back(trees[i]->getRootNode());
    return nodes;
}

void Environment::swapModels()
{
    swapped_models = !swapped_models;
}

bool Environment::areModelsSwapped()
{
    return swapped_models;
}

bool Environment::makeBestMove()
{
    Node* node = getCurrentNode()->absBestChild();
    if (node)
        return makeMove(node->getParentAction());

    Log::log(LogLevel::WARNING, "Get absBestChild failed in makeBestMove", "ENVIRONMENT");
    return false;
}

std::vector<index_t> Environment::getUntriedActions()
{

    return getCurrentNode()->getUntriedActions();
}

Node* Environment::policy()
{
    // If only 1 tree always call policy on 1.
    return trees[next_color * (trees[1] != nullptr)]->policy();
}

std::vector<std::tuple<Node*, bool>> Environment::getNetworkQueue()
{
    std::vector<std::tuple<Node*, bool>> queue;

    // Get vector of queue with network ID
    for (int i = 0; i < 2; i++)
    {
        if (trees[i] != nullptr)
        {
            for (Node* node : trees[i]->getNetworkQueue())
                if (swapped_models)
                    queue.push_back(std::tuple<Node*, bool>(node, !i));
                else
                    queue.push_back(std::tuple<Node*, bool>(node, i));
        }
    }

    return queue;
}

bool Environment::clearNetworkQueue()
{
    bool success = true;
    for (int i = 0; i < 2; i++)
        if (trees[i] != nullptr)
            if (!trees[i]->clearNetworkQueue())
                success = false;
    return success;
}

void Environment::forceClearNetworkQueue()
{
    for (int i = 0; i < 2; i++)
        if (trees[i] != nullptr)
            trees[i]->forceClearNetworkQueue();
}

Node* Environment::getCurrentNode()
{
    return trees[next_color * (trees[1] != nullptr)]->getCurrentNode();
}

void Environment::collapseEnvironment()
{
    for (int i = 0; i < 2; i++)
        if (trees[i] != nullptr)
            trees[i]->collapseTree();
}

Node* Environment::getOpposingNode()
{
    if (trees[1] == nullptr) 
        return nullptr;
    return trees[!next_color]->getCurrentNode();
}

void Environment::freeMemory()
{
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
            trees[i]->clean(); 
}

std::string Environment::toString()
{
    return getCurrentNode()->state->toString();
}

std::string Environment::toString(uint8_t depth)
{
    return Node::sliceNodeHistory(getCurrentNode(), depth);
}

bool Environment::isTerminal()
{
    return getCurrentNode()->state->isTerminal();
}

StateColor Environment::getNextColor()
{
    if (next_color)
        return StateColor::WHITE;
    return StateColor::BLACK;
}

StateResult Environment::getResult()
{
    return getCurrentNode()->getResult();
}