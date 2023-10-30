#include "Environment.h"

Environment::Environment()
    : next_color(false), swapped_models(false)
{
    for (int i = 0; i < 2; i++)
        trees[i] = new Tree();
}

Environment::~Environment()
{
    // Delete trees
    for (int i = 0; i < 2; i ++)
        delete trees[i];
}

bool Environment::makeMove(uint16_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}

void Environment::swapModels()
{
    swapped_models = !swapped_models;
}

bool Environment::areModelsSwapped()
{
    return swapped_models;
}

bool Environment::makeMove(uint8_t x, uint8_t y)
{
    // Update all existing trees
    for (int i = 0; i < 2; i ++)
    {
        // Check if move is valid
        bool successfull = trees[i]->makeMove(x, y);
        if (!successfull)
            return false;
    }

    next_color = !next_color;
    return true;
}

bool Environment::makeBestMove()
{
    Node* node = getCurrentNode()->absBestChild();
    if (node)
        return makeMove(node->parent_action);
    
    std::cout << "[Env][E]: Get absBestChild failed in makeBestMove" << std::endl;
    return false;
}

std::deque<uint16_t> Environment::getUntriedActions()
{
    return getCurrentNode()->untried_actions;
}

Node* Environment::policy()
{
    return trees[next_color]->policy();
}

std::vector<std::tuple<Node*, bool>> Environment::getNetworkQueue()
{
    std::vector<std::tuple<Node*, bool>> queue;

    // Get vector of queue with network ID
    for (int i = 0; i < 2; i++)
        for (Node* node : trees[i]->getNetworkQueue())
            if (swapped_models)
                queue.push_back(std::tuple<Node*, bool>(node, !i));
            else
                queue.push_back(std::tuple<Node*, bool>(node, i));

    return queue;
}

bool Environment::clearNetworkQueue()
{
    bool success = true;
    for (int i = 0; i < 2; i++)
        if (!trees[i]->clearNetworkQueue())
            success = false;
    return success;
}

Node* Environment::getCurrentNode()
{
    return trees[next_color]->getCurrentNode();
}

Node* Environment::getOpposingNode()
{
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

bool Environment::getNextColor()
{
    return next_color;
}

uint8_t Environment::getResult()
{
    return getCurrentNode()->getResult();
}