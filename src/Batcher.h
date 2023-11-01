#include "Config.h"
#include "Environment.h"
#include "Model.h"
#include "Node.h"

/*
Host class for the entire selfplay.
Automatically manages model calls and batches to improve performance wherever possible.
*/

class Batcher
{
public:
    Batcher(int environments, Model* NN_Black, Model* NN_White);
    Batcher(int environments, Model* only_model);
    ~Batcher();

    // Initializes all trees with their respective models
    void init();

    Environment* getEnvironment(uint32_t index);
    Node* getNode(uint32_t index);

    // Are all environments terminal
    bool isTerminal();

    // Runs MCTS policy loop n times
    void runSimulations(uint32_t simulations);

    void makeBestMoves();
    void makeRandomMoves(int amount);

    // Clears all nodes flagged for deletion
    void freeMemory();

    // Display
    std::string toString();
    // Limit max env output
    std::string toString(int max_envs);
    // Outputs a distribution for each env
    std::string toStringDist(const std::initializer_list<std::string> distributions);

    // -1 is black
    float averageWinner();

    // Evaluate models against another, use this without calling any env modifying functions before
    // Swaps models in every second environment
    void swapModels();

    // Generates duplicate envs with mirrored models. Lets them duel and outputs the win distribution between them
    float duelModels(int random_moves, int simulations);

    // Get nodes for retraining
    void storeData(std::string Path);

private:
    // Clear up all network queues
    // Should never be a need to call manually
    void runNetwork();

    // Clears non_terminal_environments of terminals
    void updateNonTerminal();

    std::vector<Environment*> environments;
    std::vector<Environment*> non_terminal_environments;
    Model* models[2] = {nullptr, nullptr};
};