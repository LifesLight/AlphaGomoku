#include "Environment.h"

float* Environment::log_table = new float[MaxSimulations];
bool Environment::is_initialized = false;

void Environment::initialize()
{
    // Init Log Table
    std::memset(log_table, 0, MaxSimulations * sizeof(float));
    for (uint32_t i = 1; i < MaxSimulations; i++)
        log_table[i] = std::log(i);

    // Set log table
    Node::setLogTable(log_table);

    is_initialized = true; 
    std::cout << "[Env]: Initialized" << std::endl;
}

Environment::Environment(Model* NNB, Model* NNW)
    : next_color(false)
{
    if (!is_initialized)
    {
        std::cout << "[Env][E]: Tried to construct a environment before calling initialize" << std::endl;
    }

    if (NNB == nullptr && NNW == nullptr)
    {
        std::cout << "[Env][E]: Environment needs at least 1 tree" << std::endl;
    }

    if (NNB != nullptr)
    {
        trees[0] = new Tree(NNB);
    }

    if (NNW != nullptr)
    {
        trees[1] = new Tree(NNW);
    }
}

bool Environment::isReady()
{
    bool ready = true;
    for (int i = 0; i < 2; i++)
        if (trees[i] != nullptr)
            if (!trees[i]->isReady())
                ready = false;
    return ready;
}

std::vector<Node*> Environment::getNetworkQueue()
{
    std::vector<Node*> queue;
    for (int i = 0; i < 2; i++)
    {
        if (trees[i] != nullptr)
        {
            for (Node* node : trees[i]->getNetworkQueue())
                queue.push_back(node);
            trees[i]->clearNetworkQueue();
        }
    }
    return queue;
}

void Environment::deinitialize()
{
    delete[] log_table;
    is_initialized = false;
}

Environment::~Environment()
{
    // Delete trees
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
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
        if (trees[i] != nullptr)
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
    if (trees[next_color] == nullptr)
    {
        std::cout << "[Env]: Error, tried to simulate for non AI player" << std::endl;
        return;
    }
    
    trees[next_color]->simulationStep();
}

Node* Environment::getAnyCurrentNode()
{
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
            return trees[i]->getCurrentNode();
    return nullptr;
}

void Environment::freeMemory()
{
    for (int i = 0; i < 2; i ++)
        if (trees[i] != nullptr)
            trees[i]->clean(); 
}

std::string Environment::toString()
{
    return getAnyCurrentNode()->state->toString();
}

std::string Environment::toString(uint8_t depth)
{
    Gamestate current_gamestate(getAnyCurrentNode());
    return current_gamestate.sliceToString(depth);
}

bool Environment::isFinished()
{
    return getAnyCurrentNode()->state->isTerminal();
}

Node* Environment::getPlayingNode()
{
    if (trees[next_color] == nullptr)
    {
        std::cout << "[Env][E]: Tried to getPlayingNode of non tree agent" << std::endl;
        return nullptr;
    }
    return trees[next_color]->getCurrentNode();
}

Node* Environment::getPlayedNode()
{
    if (trees[!next_color] == nullptr)
    {
        std::cout << "[Env][E]: Tried to getPlayedNode of non tree agent" << std::endl;
        return nullptr;
    }
    return trees[!next_color]->getCurrentNode();
}

bool Environment::getNextColor()
{
    return next_color;
}