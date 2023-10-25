#include "Config.h"
#include "Environment.h"
#include "Model.h"
#include "Node.h"

// This class is a wrapper around environments to allow simple parallelization of selfplay
class Batcher
{
public:
    Batcher(int environments, Model* NN_Black, Model* NN_White);
    ~Batcher();
    // Clear up all network queues
    void runNetwork();

    Environment* getEnvironment(uint32_t index);
    Node* getNode(uint32_t index);

    void runSimulations(uint32_t simulations);
    void makeBestMoves();
    void makeRandomMoves();
    void makeRandomMoves(int amount);

    void freeMemory();

private:
    std::vector<Environment*> environments;
    Model* models[2];
};