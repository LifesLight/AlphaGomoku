#include "Batcher.h"

Batcher::Batcher(int environment_count, Model* NNB, Model* NNW)
{
    // Store models
    models[0] = NNB;
    models[1] = NNW;

    // Super simple, needs randomization for inital gamestates
    for (int i = 0; i < environment_count; i++)
        environments.push_back(new Environment());

    runNetwork();
}

Batcher::~Batcher()
{
    for (Environment* env : environments)
        delete env;
}

void Batcher::runNetwork()
{
    // Accumilate Nodes per model
    std::vector<Node*> nodes[2];
    for (Environment* env : environments)
    {
        for (std::tuple<Node*, bool> node : env->getNetworkQueue())
            nodes[std::get<1>(node)].push_back(std::get<0>(node));
        env->clearNetworkQueue();
    }


    // Output of each model
    std::tuple<torch::Tensor, torch::Tensor> model_outputs[2];
    for (int i = 0; i < 2; i++)
    {
        // If no nodes, skip model call
        if (nodes[i].size() == 0)
            continue;

        // Convert to tensor
        uint32_t batch_size = nodes[i].size();
        torch::Tensor model_input = torch::empty({batch_size, HistoryDepth + 1, BoardSize, BoardSize}, torch::kFloat32);
        for (int index = 0; index < batch_size; index++)
        {
            torch::Tensor tensor = Node::nodeToGamestate(nodes[i][index]);
            model_input[index] = tensor;
        }

        // Run model
        model_outputs[i] = models[i]->forward(model_input);
    }

    // Assign output to node
    for (int ii = 0; ii < 2; ii++)
    {
        torch::Tensor policy_data = std::get<0>(model_outputs[ii]);
        torch::Tensor value_data = std::get<1>(model_outputs[ii]);

        for (int i = 0; i < nodes[ii].size(); i++)
        {
            std::tuple input = std::tuple<torch::Tensor, torch::Tensor>(policy_data[i], value_data[i]);
            nodes[ii][i]->setModelOutput(input);
        }
    }
}


void Batcher::runSimulations(uint32_t sim_count)
{
    // Simulation loop / MCTS loop
    uint32_t env_count = environments.size();

    std::vector<Node*> simulation_nodes;
    simulation_nodes.reserve(env_count);

    for (uint32_t sim_step = 0; sim_step < sim_count; sim_step++)
    {
        // Run policy on all envs
        for (uint32_t i = 0; i < env_count; i++)
        {
            Environment* env = environments[i];
            simulation_nodes.push_back(env->policy());
        }

        // Run network for all envs
        runNetwork();

        // Backprop all envs
        for (uint32_t i = 0; i < env_count; i++)
        {
            Environment* env = environments[i];
            Node* current_node = env->getCurrentNode();
            Node* sim_node = simulation_nodes[i];
            sim_node->backpropagate(sim_node->evaluation, current_node);
        } 

        simulation_nodes.clear();

        std::cout << "." << std::flush;
    }

    std::cout << std::endl;
}

Environment* Batcher::getEnvironment(uint32_t index)
{
    if (index < environments.size())
        return environments[index];
    
    std::cout << "Tried to get environment with out of bounds index" << std::endl;
    return nullptr;
}

Node* Batcher::getNode(uint32_t index)
{
    if (index < environments.size())
        return environments[index]->getCurrentNode();
    
    std::cout << "Tried to get node with out of bounds index" << std::endl;
    return nullptr;
}

void Batcher::freeMemory()
{
    for (Environment* env : environments)
        env->freeMemory();
}

void Batcher::makeBestMoves()
{
    for (Environment* env : environments)
        env->makeBestMove();
}

void Batcher::makeRandomMoves()
{
    for (Environment* env : environments)
        env->makeRandomMove();
}