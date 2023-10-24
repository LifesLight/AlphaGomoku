#include "Config.h"
#include "Environment.h"
#include "Model.h"
#include "Node.h"
#include "Gamestate.h"

// This class is a wrapper around environments to allow simple parallelization of selfplay
class Batcher
{
public:
    Batcher(int environments, Model* NN_Black, Model* NN_White);

    Environment* getEnvironment(uint32_t index);
    Node* getNode(uint32_t index);

    void runSimulations(uint32_t simulations);

private:
    std::vector<Environment*> environments;
    Model* models[2];

    void runNetwork();
};