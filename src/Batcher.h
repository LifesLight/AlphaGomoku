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
    // Limit max env output
    std::string toString(int max_envs);
    // Outputs a distribution for each env
    std::string toStringDist(const std::initializer_list<std::string> distributions);

    // Get winner of self play from -1 to 1
    // -1 is black
    float averageWinner();


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