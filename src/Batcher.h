#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/

#include "Config.h"
#include "Environment.h"
#include "Model.h"
#include "Node.h"
#include "Datapoint.h"
#include "Storage.h"
#include "Log.h"
#include "TreeVisualizer.h"

/*
Host class for the entire selfplay.
Automatically manages model calls and batches to improve performance wherever possible.
Is also managing multi threading since each environment is independent from ever other one this is very easy
*/

// Data for cross thread opperations

struct GCPData
{
    std::vector<std::atomic<int>*> starts;
    std::vector<std::atomic<int>*> ends;
    std::vector<std::atomic<bool>*> waits;
    std::vector<std::atomic<bool>*> running;
    std::vector<Node*>* input;

    torch::Tensor* target;
    torch::ScalarType dtype;

    // Syncing
    std::vector<std::mutex*> mutex;
    std::vector<std::condition_variable*> cv;
    std::mutex* finished_mutex;
    std::condition_variable* finished_cv;

    GCPData(int threads)
    {
        finished_mutex = new std::mutex();
        finished_cv = new std::condition_variable();

        for (int i = 0; i < threads; i++)
        {
            starts.push_back(new std::atomic<int>(0));
            ends.push_back(new std::atomic<int>(0));
            waits.push_back(new std::atomic<bool>(0));
            running.push_back(new std::atomic<bool>(1));
            mutex.push_back(new std::mutex());
            cv.push_back(new std::condition_variable());
        }
    }

    ~GCPData()
    {
        delete finished_mutex;
        delete finished_cv;

        for (int i = 0; i < int(starts.size()); i++)
        {
            delete starts[i];
            delete ends[i];
            delete waits[i];
            delete running[i];
            delete mutex[i];
            delete cv[i];
        }
    }
};

struct SIMData
{
    std::vector<std::atomic<int>*> starts;
    std::vector<std::atomic<int>*> ends;
    std::vector<std::atomic<bool>*> waits;
    std::vector<std::atomic<bool>*> running;
    std::vector<Environment*>* input;

    // Syncing
    std::vector<std::mutex*> mutex;
    std::vector<std::condition_variable*> cv;
    std::mutex* finished_mutex;
    std::condition_variable* finished_cv;

    SIMData(int threads)
    {
        finished_mutex = new std::mutex();
        finished_cv = new std::condition_variable();

        for (int i = 0; i < threads; i++)
        {
            starts.push_back(new std::atomic<int>(0));
            ends.push_back(new std::atomic<int>(0));
            waits.push_back(new std::atomic<bool>(0));
            running.push_back(new std::atomic<bool>(1));
            mutex.push_back(new std::mutex());
            cv.push_back(new std::condition_variable());
        }
    }

    ~SIMData()
    {
        delete finished_mutex;
        delete finished_cv;

        for (int i = 0; i < int(starts.size()); i++)
        {
            delete starts[i];
            delete ends[i];
            delete waits[i];
            delete running[i];
            delete mutex[i];
            delete cv[i];
        }
    }
};

class Batcher
{
public:
    Batcher(int environments, Model* NN_Black, Model* NN_White);
    Batcher(int environments, Model* only_model);
    ~Batcher();

    Environment* getEnvironment(int index);
    Node* getNode(int index);

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
    // Run policy logic on one env
    static void runPolicy(Environment* env);

    // Clear up all network queues
    // You should never need to call it manually
    void runNetwork();

    // Clears non_terminal_environments of terminals
    void updateNonTerminal();

    // Play until batcher is terminal
    void runGameloop();
    void renderEnvsHelper(bool force_render);

    std::vector<Environment*> environments;
    std::vector<Environment*> non_terminal_environments;
    Model* models[2];

    // --------- Threading ---------
    // Determine thread counts and start
    void init_threads();
    // Create the worker threads
    void start_gcp(int threads);
    void start_sim(int threads);
    // Threaded functions
    void convertNodesToGamestates(torch::Tensor& target, std::vector<Node*>* nodes, torch::ScalarType dtype);
    void runSimulationsOnEnvironments(std::vector<Environment*>* envs, int simulations);
    // Helper
    static void gcp_worker(GCPData* data, int id);
    static void sim_worker(SIMData* data, int id);
    // Helper Variables
    std::vector<std::thread*> gcp;
    std::vector<std::thread*> sim;
    GCPData* gcp_data;
    SIMData* sim_data;

    void outputTree(Node* root, int envid);

    // Tree viz
    int tree_viz_id;
    std::mt19937* rng;
};