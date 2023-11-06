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
        Environment* env = new Environment(true);
        environments.push_back(env);

        // new envs are assumed non terminal
        non_terminal_environments.push_back(env);
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Created batcher with " << environment_count << " dual tree env(s)" << std::endl;
}

Batcher::Batcher(int environment_count, Model* only_model)
{
    // Store model
    models[0] = only_model;

    environments.reserve(environment_count);
    non_terminal_environments.reserve(environment_count);

    // Super simple, needs randomization for inital gamestates
    for (int i = 0; i < environment_count; i++)
    {
        Environment* env = new Environment(false);
        environments.push_back(env);

        // new envs are assumed non terminal
        non_terminal_environments.push_back(env);
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Created batcher with " << environment_count << " single tree env(s)" << std::endl;
}

bool Batcher::getNextColor()
{
    // Just use first env since all have same next color
    return getNode(0)->getNextColor();
}

bool Batcher::getNextModelIndex(Environment* env)
{
    int model_index = getNextColor();
    model_index = (model_index * (models[1] != nullptr));
    if (env->areModelsSwapped())
        model_index = !model_index;
    return model_index;
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
    std::vector<Environment*> new_non_terminal;
    for (Environment* env : non_terminal_environments)
        if (!env->isTerminal())
            new_non_terminal.push_back(env);

    if (Utils::checkEnv("LOGGING", "INFO"))
        if (new_non_terminal.size() != non_terminal_environments.size())
            std::cout << "[Batcher][I]: " << non_terminal_environments.size() - new_non_terminal.size() << 
                            " env(s) turned terminal (" << new_non_terminal.size() << "/" << environments.size() << ")" <<  std::endl;

    non_terminal_environments = new_non_terminal;
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
        // If only 1 model run either case over same model
        model_outputs[model_index] = models[model_index * (models[1] != nullptr)]->forward(model_input);
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

void Batcher::runSimulationsOnEnvironments(std::vector<Environment*> envs, int simulations)
{
    for (int sim_step = 0; sim_step < simulations; sim_step++)
    {
        // Run policy on all envs
        for (Environment* env : envs)
        {
            Node* selected = env->policy();
            // If node has network data it wont be auto backpropagated so we manually do it again
            if (selected->getNetworkStatus())
                selected->callBackpropagate();
        }

        // Run network for all envs
        runNetwork();
    }
}

void Batcher::runSimulations()
{
    std::vector<Environment*> envsByModel[2];

    for (Environment* env : non_terminal_environments)
        envsByModel[getNextModelIndex(env)].push_back(env);

    if (Utils::checkEnv("LOGGING", "INFO"))
    {
        std::cout << "[Batcher][I]: Running simulations:" << std::endl;
        if (models[0] != nullptr && envsByModel[0].size() != 0)
            std::cout << "  " << models[0]->getName() << " on " << envsByModel[0].size() << " env(s)" << std::endl;
        if (models[1] != nullptr && envsByModel[1].size() != 0)
            std::cout << "  " << models[1]->getName() << " on " << envsByModel[1].size() << " env(s)" << std::endl;
    }
        

    for (int i = 0; i < 2; i++)
    {
        if (models[i] == nullptr)
        {
            if (envsByModel[i].size() != 0)
                std::cout << "[Batcher][W]: Skipping simulation(s) for environment(s) in uncuppler" << std::endl << std::flush;
            continue;
        }

        if (envsByModel[i].size() == 0)
            continue;

        int simulations = models[i]->getSimulations();



        runSimulationsOnEnvironments(envsByModel[i], simulations);
    }
}

void Batcher::swapModels()
{
    if (environments.size() % 2 != 0)
    {
        std::cout << "[Batcher][E]: Tried to swap every second model with uneven environment count (" << environments.size() << ")" << std::endl << std::flush;
        return;
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Swapping every second envs model" << std::endl;

    // Invert models on every second env
    for (int i = 0; i < environments.size(); i += 2)
        environments[i]->swapModels();   
}

void Batcher::runGameloop(int simulations)
{
    // Gameplay Loop
    while(!isTerminal())
    {
        runSimulations();
        makeBestMoves();

        if (Utils::checkEnv("RENDER_ENVS", "TRUE"))
        {
            int count;
            try
            {
                count = std::stoi(Utils::getEnv("RENDER_ENVS_COUNT"));
            }
            catch(const std::exception& e)
            {
                count = 2;
            }
            
            if (Utils::checkEnv("RENDER_ANALYTICS", "TRUE"))
                std::cout << toStringDist({"VISITS", "POLICY", "VALUE", "MEAN"}, count) << std::endl;
            else
                std::cout << toString(count) << std::endl;
        }
    }
}

float Batcher::duelModels(int random_actions, int simulations)
{
    if (environments.size() % 2 != 0)
    {
        std::cout << "[Batcher][E]: Tried to duel models with uneven environment count (" << environments.size() << ")" << std::endl << std::flush;
        return 0;
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Dueling models" << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());

    if (Utils::checkEnv("LOGGING", "INFO"))
        if (random_actions > 0)
            std::cout << "[Batcher][I]: Making " << random_actions << " mirrored random moves" << std::endl;

    for (int ii = 0; ii < random_actions; ii++)
    {
        // Itterate in steps of  2
        for (int i = 0; i < non_terminal_environments.size(); i += 2)
        {
            // Select random move for both envs
            Environment* env1 = non_terminal_environments[i];
            Environment* env2 = non_terminal_environments[i + 1];
            std::deque<index_t> untried;
            untried = env1->getUntriedActions();

            int action_count = untried.size();

            std::uniform_int_distribution<int> uni(0, action_count - 1);
            index_t action_index = untried[uni(rng)];

            // Make move for both envs
            env1->makeMove(action_index);
            env2->makeMove(action_index);
        }

        updateNonTerminal();
    }

    runNetwork();

    // Run until terminal
    runGameloop(simulations);

    int total_games = environments.size();
    int draws = 0;
    int non_draws = 0;
    int model1_wins = 0;
    int model2_wins = 0;
    float win_delta = 0.0f;

    // Calc average winner Model
    for (Environment* env : environments)
    {
        bool is_swapped = env->areModelsSwapped();
        uint8_t winner_color = env->getResult();

        if (winner_color == 2)
            continue;
        
        if (is_swapped)
            winner_color = !winner_color;

        if (winner_color == 0)
            model1_wins++;
        else
            model2_wins += 1;
    }

    non_draws = model1_wins + model2_wins;
    draws = total_games - non_draws;

    win_delta = float(model2_wins - model1_wins) / non_draws;

    if (Utils::checkEnv("LOGGING", "INFO"))
    {
        std::cout << "[Batcher][I]: Duel models result:" << std::endl;
        std::cout <<        "  Draws:       " << draws << std::endl;
        std::cout <<        "  Model1 wins: " << model1_wins << " (" << models[0]->getName() << ")" << std::endl;
        if (models[1] != nullptr)
            std::cout <<    "  Model2 wins: " << model2_wins << " (" << models[1]->getName() << ")" << std::endl;
        std::cout <<        "  Delta:       " << win_delta << " (-1 is Model1 wins)" << std::endl;
    }

    return win_delta;
}

void Batcher::selfplay(int simulations)
{
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Started selfplay" << std::endl;
    // Run until terminal
    runGameloop(simulations);
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
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Freeing memory" << std::endl;

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
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Making " << amount << " random moves" << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0; i < amount; i++)
    {
        for (Environment* env : non_terminal_environments)
        {   
            std::deque<index_t> untried = env->getUntriedActions();
            int action_count = untried.size();

            std::uniform_int_distribution<int> uni(0, action_count - 1);

            index_t action = untried[uni(rng)];
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

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Got average winner: " << delta << " (Black is negative)" << std::endl;

    return delta;
}

std::string Batcher::toString()
{
    return toString(non_terminal_environments.size());
}

std::string Batcher::toString(int max_envs)
{
    // In case to many max_envs
    if (max_envs > non_terminal_environments.size())
        max_envs = non_terminal_environments.size();

    std::stringstream output;
    for (int i = 0; i < max_envs; i++)
    {
        int real_id = 0;
        for (int ii = 0; ii < environments.size(); ii++)
        {
            if (environments[ii] == non_terminal_environments[i])
            {
                real_id = ii;
                break;
            }
        }

        output << std::endl << std::endl << " <---------- Environment: " << real_id << "---------->" << std::endl << std::endl; 
        output << non_terminal_environments[i]->toString();
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
    return toStringDist(distributions, non_terminal_environments.size());
}

std::string Batcher::toStringDist(const std::initializer_list<std::string> distributions, int amount)
{
    if (amount > non_terminal_environments.size())
        amount = non_terminal_environments.size();

    std::stringstream output;
    for (int i = 0; i < amount; i++)
    {
        Environment* env = non_terminal_environments[i];
        
        int real_id = 0;
        for (int ii = 0; ii < environments.size(); ii++)
        {
            if (environments[ii] == env)
            {
                real_id = ii;
                break;
            }
        }

        for (int i = 0; i < BoardSize; i++)
            output << "#-#-";
        
        output << std::endl << std::endl << " <---------- Environment: " << real_id << "---------->" << std::endl << std::endl; 

        output << " <--- Active Tree --->" << std::endl;
        output << " Model: " << models[env->getNextColor() * (models[1] != nullptr)]->getName() << std::endl;
        output << Node::analytics(env->getCurrentNode(), distributions);
        output << std::endl;

        // Need to check if exists before since possibility of single tree
        Node* opposing_node = env->getOpposingNode();
        if (opposing_node)
        {
            output << " <---- Cold Tree ---->" << std::endl;
            output << " Model: " << models[!env->getNextColor() * (models[1] != nullptr)]->getName() << std::endl;
            output << Node::analytics(opposing_node, distributions);
            output << std::endl;
        }

    }

    return output.str();   
}

void nodeCrawler(std::vector<Datapoint>& datapoints, Node* node, uint8_t winner)
{
    // Ignore non fully explored nodes
    if (node->untried_actions.size() == 0)
    {
        Datapoint data;
        data.moves = node->getMoveHistory();
        data.best_move = node->absBestChild()->parent_action;

        // Change to reflect if current play won
        if (winner == 2)
        {
            data.winner = winner;
        }
        else
        {
            if (node->getNextColor())
                data.winner = winner;
            else
                data.winner = !winner;
        }

        datapoints.push_back(data);
    }

    for (Node* child : node->children)
        if (!child->isTerminal())
            nodeCrawler(datapoints, child, winner);
}

void Batcher::storeData(std::string Path)
{
    int total_nodes = 0;
    for (Environment* env : environments)
        total_nodes += env->getNodeCount();
    
    std::vector<Datapoint> datapoints;
    // Too much since not all nodes are full expanded but whatever :)
    datapoints.reserve(total_nodes);

    // Get all nodes which are fully expanded and convert them into datapoints
    for (Environment* env : environments)
    {
        uint8_t winner = env->getResult();
        for (Node* root_node : env->getRootNodes())
            nodeCrawler(datapoints, root_node, winner);
    }

    Storage interface = Storage(Path);
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Storing " << datapoints.size() << " datapoints to: " << Path << std::endl;

    for (Datapoint data : datapoints)
        interface.storeDatapoint(data);
    interface.applyChanges();
}