#include "Batcher.h"

Batcher::Batcher(int environment_count, Model* NNB, Model* NNW)
{
    // Store models
    models[0] = NNB;
    models[1] = NNW;

    environments.reserve(environment_count);
    non_terminal_environments.reserve(environment_count);

    // Super simple, needs randomization for inital gamestates
    for (int i = 0; i < environment_count; i++)
    {
        Environment* env = new Environment();
        environments.push_back(env);

        // new envs are assumed non terminal
        non_terminal_environments.push_back(env);
    }
}

Batcher::~Batcher()
{
    for (Environment* env : environments)
        delete env;
}

void Batcher::init()
{
    runNetwork();
}

bool Batcher::isTerminal()
{
    if (non_terminal_environments.size() == 0)
        return true;
    return false;
}

void Batcher::updateNonTerminal()
{
    non_terminal_environments.clear();
    for (Environment* env : environments)
        if (!env->isTerminal())
            non_terminal_environments.push_back(env);
}

void toGamestateWorker(int index, torch::Tensor& model_input, std::vector<Node*>& nodes) 
{

}

void Batcher::runNetwork()
{
    torch::TensorOptions default_tensor_options = torch::TensorOptions().device(TorchDevice).dtype(torch::kFloat32);
    
    // Accumilate Nodes per model
    std::vector<Node*> nodes[2];
    for (Environment* env : environments)
        for (std::tuple<Node*, bool> node : env->getNetworkQueue())
            nodes[std::get<1>(node)].push_back(std::get<0>(node));

    // Output of each model
    std::tuple<torch::Tensor, torch::Tensor> model_outputs[2];
    for (int model_index = 0; model_index < 2; model_index++)
    {
        // If no nodes, skip model call
        if (nodes[model_index].size() == 0)
            continue;

        // Convert to tensor
        uint32_t batch_size = nodes[model_index].size();
        torch::Tensor model_input = torch::empty({batch_size, HistoryDepth + 1, BoardSize, BoardSize}, default_tensor_options);

        for (int index = 0; index < batch_size; index++) 
        {
            model_input[index] = Node::nodeToGamestate(nodes[model_index][index]);
        }

        // Run model
        model_outputs[model_index] = models[model_index]->forward(model_input);
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

    // Clear network queue
    for (Environment* env : environments)
    {
        bool success = env->clearNetworkQueue();
        if (!success)
            std::cout << "[Batcher][W]: Network queue could not be cleared (Nodes without Netdata remaining)" << std::endl;
    }
}

// Only run on non terminal environments
void Batcher::runSimulations(uint32_t sim_count)
{
    // Simulation loop / MCTS loop
    uint32_t env_count = non_terminal_environments.size();

    std::vector<Node*> simulation_nodes;
    simulation_nodes.reserve(env_count);

    for (uint32_t sim_step = 0; sim_step < sim_count; sim_step++)
    {
        // Run policy on all envs
        for (uint32_t i = 0; i < env_count; i++)
        {
            Environment* env = non_terminal_environments[i];
            simulation_nodes.push_back(env->policy());
        }

        // Run network for all envs
        runNetwork();

        simulation_nodes.clear();
    }
}

void Batcher::swapModels()
{
    if (environments.size() % 2 != 0)
    {
        std::cout << "[Batcher][E]: Tried to swap every second model with uneven environment count (" << environments.size() << ")" << std::endl << std::flush;
        return;
    }

    // Invert models on every second env
    for (int i = 0; i < environments.size(); i += 2)
        environments[i]->swapModels();   
}

float Batcher::duelModels(int random_actions, int simulations)
{
    if (environments.size() % 2 != 0)
    {
        std::cout << "[Batcher][E]: Tried to duel models with uneven environment count (" << environments.size() << ")" << std::endl << std::flush;
        return 0;
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int ii = 0; ii < random_actions; ii++)
    {
        // Itterate in steps of  2
        for (int i = 0; i < non_terminal_environments.size(); i += 2)
        {
            // Select random move for both envs
            Environment* env1 = non_terminal_environments[i];
            Environment* env2 = non_terminal_environments[i + 1];
            std::deque<uint16_t> untried;
            untried = env1->getUntriedActions();

            int action_count = untried.size();

            std::uniform_int_distribution<int> uni(0, action_count - 1);
            uint16_t action_index = untried[uni(rng)];

            // Make move for both envs
            env1->makeMove(action_index);
            env2->makeMove(action_index);
        }

        updateNonTerminal();
    }

    runNetwork();
    
    // Gameplay Loop
    while(!isTerminal())
    {
        runSimulations(simulations);
        makeBestMoves();
        std::cout << toString(2) << std::endl << std::flush;
    }

    int non_draws = 0;
    float win_delta = 0.0f;

    // Calc average winner Model
    for (Environment* env : environments)
    {
        bool is_swapped = env->areModelsSwapped();
        uint8_t winner_color = env->getResult();

        if (winner_color == 2)
            continue;

        non_draws++;
        
        if (is_swapped)
            winner_color = !winner_color;

        if (winner_color == 0)
            win_delta -= 1;
        else
            win_delta += 1;
    }

    win_delta /= non_draws;
    return win_delta;
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
    for (Environment* env : non_terminal_environments)
        env->makeBestMove();

    updateNonTerminal();
    runNetwork();
}

void Batcher::makeRandomMoves(int amount)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0; i < amount; i++)
    {
        for (Environment* env : non_terminal_environments)
        {   
            std::deque<uint16_t> untried = env->getUntriedActions();
            int action_count = untried.size();

            std::uniform_int_distribution<int> uni(0, action_count - 1);

            uint16_t action = untried[uni(rng)];
            env->makeMove(action);
        }

        updateNonTerminal();
    }

    runNetwork();
}

float Batcher::averageWinner()
{
    if (non_terminal_environments.size() != 0)
    {
        std::cout << "[Batcher][W]: Got average winner before all environments finished playing" << std::endl;
    }
    
    int total = environments.size() - non_terminal_environments.size();
    float delta = 0;

    for (Environment* env : environments)
    {
        // Skip if env is terminal
        if (!env->isTerminal())
            continue;

        uint8_t result = env->getResult();

        if      (result == 0)
            delta -= 1;
        else if (result == 1)
            delta += 1;
    }

    delta = delta / total;

    return delta;
}

std::string Batcher::toString()
{
    return toString(environments.size());
}

std::string Batcher::toString(int max_envs)
{
    // In case to many max_envs
    if (max_envs > environments.size())
        max_envs = environments.size();

    std::stringstream output;
    for (int i = 0; i < max_envs; i++)
    {
        output << std::endl << std::endl << " <---------- Environment: " << i << "---------->" << std::endl << std::endl; 
        output << environments[i]->toString();
    }

    // Show that not all envs are displayed
    if (max_envs < environments.size())
    {
        if (non_terminal_environments.size() > 0)
            output << std::endl << "(" << std::max(int(non_terminal_environments.size()) - max_envs, 0) << "/" << environments.size() - max_envs << ") ...." << std::endl;
        else
            output << std::endl << "(" << environments.size() - max_envs << ") ...." << std::endl;
    }

    return output.str();
}


std::string Batcher::toStringDist(const std::initializer_list<std::string> distributions)
{
    std::stringstream output;
    for (int i = 0; i < environments.size(); i++)
    {
        for (int i = 0; i < BoardSize; i++)
            output << "#-#-";
        
        output << std::endl << std::endl << " <---------- Environment: " << i << "---------->" << std::endl << std::endl; 

        output << " <--- Active Tree --->" << std::endl;
        output << Node::analytics(environments[i]->getCurrentNode(), distributions);
        output << std::endl;

        output << " <---- Cold Tree ---->" << std::endl;
        output << Node::analytics(environments[i]->getOpposingNode(), distributions);
        output << std::endl;
    }

    return output.str();
}

std::vector<Node*> Batcher::sampleNodes(int amount)
{
    int total_nodes;

    for (Environment* env : environments)
    {
        for (Node* root_node : env->getRootNodes())
        {
            
        }
    }
}