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

int Environment::getNodeCount()
{
    int sum = 0;
    for (int i = 0; i < 2; i++)
        if (trees[i] != nullptr)
            sum += trees[i]->getNodeCount();
    return sum;
}

bool Environment::makeMove(index_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return makeMove(x, y);
}

bool Environment::makeMove(uint8_t x, uint8_t y)
{
    // Update all existing trees
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
            trees[i]->makeMove(x, y);

    next_color = !next_color;
    return true;
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
        return makeMove(node->parent_action);
    
    std::cout << "[Env][E]: Get absBestChild failed in makeBestMove" << std::endl;
    return false;
}

std::deque<index_t> Environment::getUntriedActions()
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

Node* Environment::getOpposingNode()
{
    if (trees[1] == nullptr) 
        return nullptr;
    return trees[!(next_color * (trees[1] != nullptr))]->getCurrentNode();
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