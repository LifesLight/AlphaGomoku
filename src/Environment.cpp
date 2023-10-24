#include "Environment.h"

Environment::Environment()
    : next_color(false)
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

void Environment::simulationStep()
{

}

std::vector<std::tuple<Node*, bool>> Environment::getNetworkQueue()
{
    std::vector<std::tuple<Node*, bool>> queue;

    // Get vector of queue with network ID
    for (int i = 0; i < 2; i++)
        for (Node* node : trees[i]->getNetworkQueue())
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
    Gamestate current_gamestate(getCurrentNode());
    return current_gamestate.sliceToString(depth);
}

bool Environment::isFinished()
{
    return getCurrentNode()->state->isTerminal();
}

bool Environment::getNextColor()
{
    return next_color;
}