#include "Batcher.h"

Batcher::Batcher(int environment_count, Model* NNB, Model* NNW)
    : gcp_data(nullptr), sim_data(nullptr)
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

    // Threading init
    init_threads();

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Created batcher with " << environment_count << " dual tree env(s)" << std::endl;
}

Batcher::Batcher(int environment_count, Model* only_model)
    : gcp_data(nullptr), sim_data(nullptr)
{
    // Store model
    models[0] = only_model;
    models[1] = nullptr;

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

    // Threading init
    init_threads();

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Created batcher with " << environment_count << " single tree env(s)" << std::endl;
}

Batcher::~Batcher()
{
    if (Utils::checkEnv("LOGGING", "INFO"))
        ForcePrint("[Batcher][I]: Started deconstructing batcher ... ");

    for (Environment* env : environments)
        delete env;

    // Call to threads to finish
    for (int i = 0; i < int(gcp.size()); i++)
    {
        gcp_data->running[i]->store(false);
        gcp_data->cv[i]->notify_one();
    }

    for (int i = 0; i < int(sim.size()); i++)
    {
        sim_data->running[i]->store(false);
        sim_data->cv[i]->notify_one();
    }

    // Join all threads
    // TODO Better solution
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Delete threading data
    if (gcp_data)
        delete gcp_data;

    if (sim_data)
        delete sim_data;

    if (Utils::checkEnv("LOGGING", "INFO"))
        ForcePrintln("finished");
}

void Batcher::init_threads()
{
    int gcp_threads = std::max(1, std::min(Config::maxThreads(), int(environments.size() / Config::gamestatesPerThread())));
    int sim_threads = std::max(1, std::min(Config::maxThreads(), int(environments.size()) / Config::simsPerThread()));

    if (gcp_threads > 1)
    {
        start_gcp(gcp_threads);
    }

    if (sim_threads > 1)
    {
        start_sim(sim_threads);
    }
}

void Batcher::gcp_worker(GCPData* data, int id)
{
    std::unique_lock<std::mutex> lock(*data->mutex[id]);

    while (data->running[id]->load())
    {
        // Wait for signal from thread manager
        data->cv[id]->wait(lock, [&]() { return data->waits[id]->load(); });

        for (int i = data->starts[id]->load(); i < data->ends[id]->load(); i++)
        {
            (*data->target)[i] = Node::nodeToGamestate((*data->input)[i], data->dtype);
        }
        data->waits[id]->store(false);

        // Singal that worker is finished
        {
            std::lock_guard<std::mutex> finishLock(*data->finished_mutex);
            data->finished_cv->notify_one();
        }
    }
}

void Batcher::sim_worker(SIMData* data, int id)
{
    std::unique_lock<std::mutex> lock(*data->mutex[id]);

    while (data->running[id]->load())
    {
        // Wait for signal from thread manager
        data->cv[id]->wait(lock, [&]() { return data->waits[id]->load(); });

        for (int i = data->starts[id]->load(); i < data->ends[id]->load(); i++)
        {
            runPolicy((*data->input)[i]);
        }
        data->waits[id]->store(false);

        // Singal that worker is finished
        {
            std::lock_guard<std::mutex> finishLock(*data->finished_mutex);
            data->finished_cv->notify_one();
        }
    }
}

void Batcher::start_gcp(int threads)
{
    gcp_data = new GCPData(threads);

    for (int i = 0; i < threads; i++)
    {
        std::thread* worker = new std::thread(gcp_worker, gcp_data, i);
        gcp.push_back(worker);
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Started " << threads << " GCP thread(s)" << std::endl;
}

void Batcher::start_sim(int threads)
{
    sim_data = new SIMData(threads);

    for (int i = 0; i < threads; i++)
    {
        std::thread* worker = new std::thread(sim_worker, sim_data, i);
        sim.push_back(worker);
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Started " << threads << " SIM thread(s)" << std::endl;
}

bool Batcher::getNextModelIndex(Environment* env)
{
    bool model_index = env->getNextColor();

    if (env->areModelsSwapped())
        model_index = !model_index;

    if (models[1] == nullptr)
        model_index = 0;

    return model_index;
}

bool Batcher::isTerminal()
{
    if (non_terminal_environments.size() == 0)
        return true;
    return false;
}

void Batcher::runPolicy(Environment* env)
{
    Node* selected = env->policy();
    // If node already with netdata implies it didnt get backpropagated so we manually call it again
    if (selected->getNetworkStatus())
        selected->callBackpropagate();
}

void Batcher::updateNonTerminal()
{
    std::vector<Environment*> new_non_terminal;
    for (Environment* env : non_terminal_environments)
    {
        if (!env->isTerminal())
            new_non_terminal.push_back(env);
        else
            env->collapseEnvironment();
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        if (new_non_terminal.size() != non_terminal_environments.size())
            std::cout << "[Batcher][I]: " << non_terminal_environments.size() - new_non_terminal.size() << 
                            " env(s) turned terminal (" << new_non_terminal.size() << "/" << environments.size() << ")" <<  std::endl;

    non_terminal_environments = new_non_terminal;
}

void Batcher::runNetwork()
{
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    // Accumilate Nodes per model
    std::vector<Node*> nodes[2];
    for (Environment* env : non_terminal_environments)
        for (std::tuple<Node*, bool> node : env->getNetworkQueue())
            nodes[std::get<1>(node)].push_back(std::get<0>(node));

    // Output of each model
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> model_outputs[2];
    for (int model_index = 0; model_index < 2; model_index++)
    {
        // If no nodes, skip model call
        if (nodes[model_index].size() == 0)
            continue;

        // Save to index into models
        int checked_model_index = model_index * (models[1] != nullptr);

        // Maybe models with different precs
        torch::ScalarType dtype = models[checked_model_index]->getPrec();
        torch::Device device = models[checked_model_index]->getDevice();
        torch::TensorOptions default_tensor_options = torch::TensorOptions().device(Config::torchHostDevice()).dtype(dtype).requires_grad(false);

        int element_count = nodes[model_index].size();
        // Compute gamestates with multithreading
        torch::Tensor gamestates;
        convertNodesToGamestates(gamestates, &nodes[model_index], dtype);

        // Batchsize limiting to not explode memory
        model_outputs[model_index].reserve(element_count / Config::maxBatchsize());
        int processed_element_count = 0;
        while (processed_element_count != element_count)
        {
            int unprocessed_count = element_count - processed_element_count;
            int batch_size = std::min(unprocessed_count, Config::maxBatchsize());

            // Convert to tensor
            // Init tensor on CPU
            torch::Tensor model_input = torch::empty({batch_size, Config::historyDepth() + 1, BoardSize, BoardSize}, default_tensor_options);
            model_input = gamestates.index({torch::indexing::Slice(processed_element_count, processed_element_count + batch_size)});

            // Move to device for inference
            model_input = model_input.to(device);

            // Run model
            // If only 1 model run either case over same model
            auto output = models[checked_model_index]->forward(model_input);
            model_outputs[model_index].push_back(output);
            processed_element_count += batch_size;
        }
    }

    // Assign output to node
    for (int ii = 0; ii < 2; ii++)
    {
        int index = 0;
        for (std::tuple<torch::Tensor, torch::Tensor> chunk : model_outputs[ii])
        {
            torch::Tensor policy_data = std::get<0>(chunk);
            torch::Tensor value_data = std::get<1>(chunk);

            for (int i = 0; i < policy_data.size(0); i++)
            {
                nodes[ii][index]->setModelOutput(policy_data[i], value_data[i]);
                index++;
            }
        }
    }

    // Clear network queue
    for (Environment* env : non_terminal_environments)
    {
        bool success = env->clearNetworkQueue();
        if (!success)
            std::cout << "[Batcher][W]: Network queue could not be cleared (Nodes without Netdata remaining)" << std::endl;
    }
}

void Batcher::convertNodesToGamestates(torch::Tensor& target, std::vector<Node*>* nodes, torch::ScalarType dtype)
{
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    int element_count = nodes->size();

    // Compute "optimal" thread count
    int thread_count = std::max(1, element_count / Config::gamestatesPerThread());
    thread_count = std::min(thread_count, int(gcp.size()));

    // Init output tensor
    torch::TensorOptions default_tensor_options = torch::TensorOptions().device(Config::torchHostDevice()).dtype(dtype).requires_grad(false);
    target = torch::empty({element_count, Config::historyDepth() + 1, BoardSize, BoardSize}, default_tensor_options);

    // Use single if possible
    if (thread_count < 2)
    {
        for (int i = 0; i < element_count; i++)
        {
            target[i] = Node::nodeToGamestate((*nodes)[i], dtype);
        }

        return;
    }

    // Set params for Threading
    gcp_data->input = nodes;
    gcp_data->target = &target;
    gcp_data->dtype = dtype;

    // Calculate index ranges
    int batch_size = int(std::ceil(float(element_count) / thread_count));

    // Start workers
    for (int i = 0; i < thread_count; i++)
    {
        // Calc index ranges
        int start_index = i * batch_size;
        gcp_data->starts[i]->store(start_index);
        gcp_data->ends[i]->store(std::min(start_index + batch_size, element_count));
        gcp_data->waits[i]->store(true);
        gcp_data->cv[i]->notify_one();
    }

    // Wait for workers to finish
    {
        std::unique_lock<std::mutex> lock(*gcp_data->finished_mutex);
        gcp_data->finished_cv->wait(lock, [&]() {
            for (int i = 0; i < thread_count; i++) {
                if (gcp_data->waits[i]->load()) {
                    return false;
                }
            }
            return true;
        });
    }
}

void Batcher::runSimulationsOnEnvironments(std::vector<Environment*>* envs, int simulations)
{
    int element_count = envs->size();

    // Compute "optimal" thread count
    int thread_count = std::max(1, element_count / Config::simsPerThread());
    thread_count = std::min(thread_count, int(gcp.size()));

    // If thread count too low just run single threaded opperation
    if (thread_count < 2)
    {
        for (int i = 0; i < simulations; i++)
        {
            for (Environment* env : *envs)
            {
                runPolicy(env);
            }

            runNetwork();
        }

        return;
    }

    sim_data->input = envs;

    // Calculate index ranges
    int batch_size = int(std::ceil(float(element_count) / thread_count));

    for (int sim = 0; sim < simulations; sim++)
    {
        // Start workers
        for (int i = 0; i < thread_count; i++)
        {
            // Calc index ranges
            int start_index = i * batch_size;
            sim_data->starts[i]->store(start_index);
            sim_data->ends[i]->store(std::min(start_index + batch_size, element_count));
            sim_data->waits[i]->store(true);
            sim_data->cv[i]->notify_one();
        }

        // Wait for workers to finish
        {
            std::unique_lock<std::mutex> lock(*sim_data->finished_mutex);
            sim_data->finished_cv->wait(lock, [&]() {
                for (int i = 0; i < thread_count; i++) {
                    if (sim_data->waits[i]->load()) {
                        return false;
                    }
                }
                return true;
            });
        }

        runNetwork();
    }
}

// TODO make cuppled option
void Batcher::runSimulations()
{
    std::vector<Environment*> envsByModel[2];

    for (Environment* env : non_terminal_environments)
        envsByModel[getNextModelIndex(env)].push_back(env);

    if (Utils::checkEnv("LOGGING", "INFO"))
    {
        std::cout << "[Batcher][I]: Running simulation(s):" << std::endl;
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
                ForcePrintln("[Batcher][W]: Skipping simulation(s) for environment(s) in uncuppler");
            continue;
        }

        if (envsByModel[i].size() == 0)
            continue;

        int simulations = models[i]->getSimulations();
        runSimulationsOnEnvironments(&envsByModel[i], simulations);
    }
}

void Batcher::swapModels()
{
    if (environments.size() % 2 != 0)
    {
        ForcePrintln("[Batcher][E]: Tried to swap every second model with uneven environment count (" << environments.size() << ")");
        return;
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Swapping every second envs model" << std::endl;

    // Invert models on every second env
    for (int i = 0; i < int(environments.size()); i += 2)
        environments[i]->swapModels();
}

void Batcher::renderEnvsHelper()
{
    if (Utils::checkEnv("RENDER_ENVS", "TRUE"))
    {
        int count;
        try
        {
            count = std::stoi(Utils::getEnv("RENDER_ENVS_COUNT"));
        }
        catch(const std::exception& e)
        {
            count = 1;
        }
        if (Utils::checkEnv("RENDER_ANALYTICS", "TRUE"))
            std::cout << toStringDist({"VISITS", "POLICY", "VALUE", "MEAN"}, count) << std::endl;
        else
            std::cout << toString(count) << std::endl;
    }
}

void Batcher::runGameloop()
{
    // Gameplay Loop
    while(!isTerminal())
    {
        runSimulations();
        makeBestMoves();
        renderEnvsHelper();
        freeMemory();
    }
}

bool Batcher::isSingleTree()
{
    return models[1] == nullptr;
}

float Batcher::duelModels()
{
    if (environments.size() % 2 != 0)
    {
        ForcePrintln("[Batcher][E]: Tried to duel models with uneven environment count (" << environments.size() << ")");
        return 0;
    }

    if (isSingleTree())
    {
        ForcePrintln("[Batcher][E]: Tried to duel models in single tree mode (Create batcher with 2 models to fix this)");
        return 0;
    }

    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Dueling models" << std::endl;

    // Run until terminal
    runGameloop();

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

void Batcher::selfplay()
{
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Started selfplay" << std::endl;
    // Run until terminal
    runGameloop();

    // Log outcome
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Selfplay result: " << averageWinner() << " average winning color" << std::endl;
}

void Batcher::humanplay(bool human_color)
{
    runNetwork();
    // TODO: Random bug with 1 sim and human winning
    while (!isTerminal())
    {
        if (environments[0]->getNextColor() == human_color)
        {
            index_t x, y;
            Utils::keyboardCordsInput(x, y);
            if (environments[0]->makeMove(x, y))
                runNetwork();
            else
                ForcePrintln("[Batcher][E]: Failed to perfom move!");
        }
        else
        {
            runSimulations();
            makeBestMoves();
        }

        renderEnvsHelper();
        freeMemory();
    }
}

Environment* Batcher::getEnvironment(int index)
{
    if (index < int(environments.size()))
        return environments[index];

    std::cout << "Tried to get environment with out of bounds index" << std::endl;
    return nullptr;
}

Node* Batcher::getNode(int index)
{
    if (index < int(environments.size()))
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

void Batcher::makeRandomMoves(int amount, bool mirrored)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    if (mirrored)
    {
        if (environments.size() % 2 == 1)
        {
            ForcePrintln("[Batcher][E]: Tried to make mirrored random moves with uneven env count");
            return;
        }

        if (Utils::checkEnv("LOGGING", "INFO"))
            std::cout << "[Batcher][I]: Making " << amount << " mirrored random moves" << std::endl;

        for (int ii = 0; ii < amount; ii++)
        {
            // Itterate in steps of  2
            for (int i = 0; i < int(non_terminal_environments.size()); i += 2)
            {
                // Select random move for both envs
                Environment* env1 = non_terminal_environments[i];
                Environment* env2 = non_terminal_environments[i + 1];
                std::vector<index_t> untried;
                untried = env1->getUntriedActions();

                int action_count = untried.size();

                std::uniform_int_distribution<int> uni(0, action_count - 1);
                index_t action_index = untried[uni(rng)];

                // Make move for both envs
                env1->makeMove(action_index);
                env2->makeMove(action_index);
            }

            // If not last itteration force clear network queue because those nodes will never need net data
            if (ii < amount - 1)
            {
                for (Environment* env : non_terminal_environments)
                    env->forceClearNetworkQueue();
            }

            updateNonTerminal();
        }
    }
    else
    {
        if (Utils::checkEnv("LOGGING", "INFO"))
            std::cout << "[Batcher][I]: Making " << amount << " random moves" << std::endl;

        for (int i = 0; i < amount; i++)
        {
            for (Environment* env : non_terminal_environments)
            {
                std::vector<index_t> untried = env->getUntriedActions();
                int action_count = untried.size();

                std::uniform_int_distribution<int> uni(0, action_count - 1);

                index_t action = untried[uni(rng)];
                env->makeMove(action);
            }

            // If not last itteration force clear network queue because those nodes will never need net data
            if (i < amount - 1)
            {
                for (Environment* env : non_terminal_environments)
                    env->forceClearNetworkQueue();
            }

            updateNonTerminal();
        }
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
    if (max_envs > int(non_terminal_environments.size()))
        max_envs = non_terminal_environments.size();

    std::stringstream output;
    for (int i = 0; i < max_envs; i++)
    {
        int real_id = 0;
        for (int ii = 0; ii < int(environments.size()); ii++)
        {
            if (environments[ii] == non_terminal_environments[i])
            {
                real_id = ii;
                break;
            }
        }

        output << std::endl << std::endl << "        ᐊ";
        for (int i = 0; i < BoardSize; i++)
            output << "═";
        output << "╡ Environment: " << real_id << " ╞"; 

        for (int i = 0; i < BoardSize; i++)
            output << "═";
        output << "ᐅ" << std::endl << std::endl;
        output << non_terminal_environments[i]->toString();
    }

    // Show that not all envs are displayed
    if (max_envs < int(environments.size()))
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
    // Clipping to non terminal environments so we dont render "dead" envs
    if (amount > int(non_terminal_environments.size()))
        amount = int(non_terminal_environments.size());

    std::stringstream output;
    for (int i = 0; i < amount; i++)
    {
        Environment* env = non_terminal_environments[i];

        int real_id = 0;
        for (int ii = 0; ii < int(environments.size()); ii++)
        {
            if (environments[ii] == env)
            {
                real_id = ii;
                break;
            }
        }

        for (int i = 0; i < BoardSize; i++)
            output << "#-#-";

        output << std::endl << "     ᐊ";
        for (int i = 0; i < BoardSize; i++)
            output << "═";
        output << "╡ Environment: " << real_id << " ╞";
        for (int i = 0; i < BoardSize; i++)
            output << "═";
        output << "ᐅ" << std::endl << std::endl << "   ";

        for (int i = 0; i < BoardSize; i++)
            output << " ";
        output << "ᐊ";
        for (int i = 0; i < 3; i++)
            output << "═";
        output << "╡ Active Tree ╞";
        for (int i = 0; i < 3; i++)
            output << "═";
        output << "ᐅ" << std::endl << std::endl;
        output << " Model: " << models[env->getNextColor() * (models[1] != nullptr)]->getName() << std::endl;
        output << Node::analytics(env->getCurrentNode(), distributions);
        output << std::endl;

        // Need to check if exists before since possibility of single tree
        Node* opposing_node = env->getOpposingNode();
        if (opposing_node)
        {
            std::cout << "   ";
            for (int i = 0; i < BoardSize; i++)
                output << " ";
            output << "ᐊ";
            for (int i = 0; i < 4; i++)
                output << "═";
            output << "╡ Cold Tree ╞";
            for (int i = 0; i < 4; i++)
                output << "═";
            output << "ᐅ" << std::endl << std::endl;
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
    if (node->isFullyExpanded())
    {
        Datapoint data;
        data.moves = node->getMoveHistory();
        data.best_move = node->absBestChild()->getParentAction();
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
    // Datapoints to be stored
    std::vector<Datapoint> datapoints;

    // Get all nodes which are fully expanded and convert them into datapoints
    for (Environment* env : environments)
    {
        uint8_t winner = env->getResult();
        for (Node* root_node : env->getRootNodes())
        {
            nodeCrawler(datapoints, root_node, winner);
        }
    }

    Storage interface = Storage(Path);
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Batcher][I]: Storing " << datapoints.size() << " datapoints to: " << Path << std::endl;

    for (Datapoint data : datapoints)
        interface.storeDatapoint(data);
    interface.applyChanges();
}