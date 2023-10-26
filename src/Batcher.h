#include "Config.h"
#include "Environment.h"
#include "Model.h"
#include "Node.h"

// TODO Deal with terminal environments

// This class is a wrapper around environments to allow simple parallelization of selfplay
class Batcher
{
public:
    Batcher(int environments, Model* NN_Black, Model* NN_White);
    ~Batcher();

    Environment* getEnvironment(uint32_t index);
    Node* getNode(uint32_t index);

    bool isTerminal();

    void runSimulations(uint32_t simulations);

    void makeBestMoves();
    void makeRandomMoves();
    void makeRandomMoves(int amount);

    void freeMemory();

    // Display
    std::string toString();

private:
    // Clear up all network queues
    // Should never be a need to call manually
    void runNetwork();

    // Clears non_terminal_environments of terminals
    void updateNonTerminal();

    std::vector<Environment*> environments;
    std::vector<Environment*> non_terminal_environments;
    Model* models[2];
};