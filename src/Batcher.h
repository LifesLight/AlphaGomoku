#include "Config.h"
#include "Environment.h"
#include "Model.h"
#include "Node.h"
#include "Datapoint.h"
#include "Storage.h"

/*
Host class for the entire selfplay.
Automatically manages model calls and batches to improve performance wherever possible.
*/

struct GCPData
{
    std::vector<std::atomic<int>*> starts;
    std::vector<std::atomic<int>*> ends;
    std::vector<std::atomic<bool>*> waits;
    std::vector<Node*>* input;
    // Load input data here before calling workers
    torch::Tensor* target;
    torch::ScalarType dtype;

    ~GCPData()
    {
        for (int i = 0; i < starts.size(); i++)
        {
            delete starts[i];
            delete ends[i];
            delete waits[i];
        }
    }
};

class Batcher
{
public:
    Batcher(int environments, Model* NN_Black, Model* NN_White);
    Batcher(int environments, Model* only_model);
    ~Batcher();

    Environment* getEnvironment(uint32_t index);
    Node* getNode(uint32_t index);

    // Are all environments terminal
    bool isTerminal();

    // Check weather env is in single or dual tree mode
    bool isSingleTree();

    // Run simulations according to model
    void runSimulations();

    void makeBestMoves();
    void makeRandomMoves(int amount, bool mirrored);

    // Clears all nodes flagged for deletion
    void freeMemory();

    // Display
    std::string toString();
    // Limit max env output
    std::string toString(int max_envs);
    // Outputs a distribution for each env
    std::string toStringDist(const std::initializer_list<std::string> distributions);
    std::string toStringDist(const std::initializer_list<std::string> distributions, int amount);

    // -1 is black
    float averageWinner();

    // Evaluate models against another, use this without calling any env modifying functions before
    // Swaps models in every second environment
    void swapModels();

    // Generates duplicate envs with mirrored models. Lets them duel and outputs the win distribution between them
    float duelModels();

    // Play model against itself, highly recommended to use in single tree mode
    void selfplay();

    // Play as human vs model
    void humanplay(bool human_color);

    // Get nodes for retraining
    void storeData(std::string path);

    // Model that run simulations is going the called on
    bool getNextModelIndex(Environment* env);

private:
    // ------------- Multithreaded -------------
    void convertNodesToGamestates(torch::Tensor& target, std::vector<Node*>* nodes, torch::ScalarType dtype);

    // Run simulation loop on provided environments
    static void runSimulationsOnEnvironmentsWorker(std::vector<Environment*>& envs, int start_index, int end_index);
    void runSimulationsOnEnvironments(std::vector<Environment*>& envs, int simulations);

    // ------------------------------------------

    // Clear up all network queues
    // You should never need to call it manually
    void runNetwork();

    // Create the worker threads
    void initThreadpool();

    // Clears non_terminal_environments of terminals
    void updateNonTerminal();

    // Play until batcher is terminal
    void runGameloop();

    // Threading
    static void gcp_worker(GCPData* data, int id);
    // gcp = gamestate conversion pool
    std::vector<std::thread*> gcp;
    GCPData* gcp_data;

    std::vector<Environment*> environments;
    std::vector<Environment*> non_terminal_environments;
    Model* models[2];
};