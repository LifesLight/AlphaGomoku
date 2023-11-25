#include "Batcher.h"

Batcher::Batcher(int environment_count, Model* NNB, Model* NNW)
    : gcp_data(nullptr), sim_data(nullptr), tree_viz_id(0)
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

    Log::log(LogLevel::INFO, "Created batcher with " + std::to_string(environment_count) + " dual tree env(s)", "BATCHER");
}

Batcher::Batcher(int environment_count, Model* only_model)
    : gcp_data(nullptr), sim_data(nullptr), tree_viz_id(0)
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

    Log::log(LogLevel::INFO, "Created batcher with " + std::to_string(environment_count) + " single tree env(s)", "BATCHER");
}

Batcher::~Batcher()
{
    Log::log(LogLevel::INFO, "Started deconstructing batcher", "BATCHER");

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

    Log::log(LogLevel::INFO, "Finished deconstructing batcher", "BATCHER");
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

    Log::log(LogLevel::INFO, "Started " + std::to_string(threads) + " GCP thread(s)", "BATCHER");
}

void Batcher::start_sim(int threads)
{
    sim_data = new SIMData(threads);

    for (int i = 0; i < threads; i++)
    {
        std::thread* worker = new std::thread(sim_worker, sim_data, i);
        sim.push_back(worker);
    }

    Log::log(LogLevel::INFO, "Started " + std::to_string(threads) + " SIM thread(s)", "BATCHER");
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

    if (new_non_terminal.size() != non_terminal_environments.size())
        Log::log(LogLevel::INFO, "Updated non terminal envs from " + std::to_string(non_terminal_environments.size()) + " to " + std::to_string(new_non_terminal.size()), "BATCHER");

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
            Log::log(LogLevel::WARNING, "Network queue could not be cleared (Nodes without Netdata remaining)", "BATCHER");
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

void Batcher::runSimulations()
{
    std::vector<Environment*> envsByModel[2];

    for (Environment* env : non_terminal_environments)
        envsByModel[getNextModelIndex(env)].push_back(env);

    Log::log(LogLevel::INFO, "Running simulation(s):", "BATCHER");
    if (models[0] != nullptr && envsByModel[0].size() != 0)
        Log::log(LogLevel::INFO, "  " + models[0]->getName() + " on " + std::to_string(envsByModel[0].size()) + " env(s)", "BATCHER");
    if (models[1] != nullptr && envsByModel[1].size() != 0)
        Log::log(LogLevel::INFO, "  " + models[1]->getName() + " on " + std::to_string(envsByModel[1].size()) + " env(s)", "BATCHER");


    for (int i = 0; i < 2; i++)
    {
        if (models[i] == nullptr)
        {
            if (envsByModel[i].size() != 0)
                Log::log(LogLevel::WARNING, "Skipping simulation(s) fo environment(s) in uncuppler", "BATCHER");

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
        Log::log(LogLevel::ERROR, "Tried to swap models with uneven environment count (" + std::to_string(environments.size()) + ")", "BATCHER");
        return;
    }

    Log::log(LogLevel::INFO, "Swapping every second envs model", "BATCHER");

    // Invert models on every second env
    for (int i = 0; i < int(environments.size()); i += 2)
        environments[i]->swapModels();
}

void Batcher::renderEnvsHelper(bool force_render)
{
    if (Config::renderEnvs() || force_render || Config::renderAnalytics())
    {
        int count = Config::renderEnvsCount();

        if (Config::renderAnalytics())
            std::cout << toStringDist({"VISITS", "POLICY", "VALUE", "MEAN"}, count) << std::endl;
        else
            std::cout << toString(count) << std::endl;
    }
}

void Batcher::runGameloop()
{
    runNetwork();

    // Gameplay Loop
    while(!isTerminal())
    {
        runSimulations();
        if (Config::outputTrees())
        {
            int id = 0;
            Log::log(LogLevel::INFO, "Writing tree(s) to: " + Config::outputTreesPath(), "BATCHER");
            for (Environment* env : non_terminal_environments)
            {
                Node* root = env->getCurrentNode();
                outputTree(root, id);
                id++;
            }
            tree_viz_id++;
        }
        makeBestMoves();
        renderEnvsHelper(false);
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
        Log::log(LogLevel::ERROR, "Tried to duel models with uneven environment count (" + std::to_string(environments.size()) + ")", "BATCHER");
        return 0;
    }

    if (isSingleTree())
    {
        Log::log(LogLevel::ERROR, "Tried to duel models in single tree mode (Create batcher with 2 models to fix this)", "BATCHER");
        return 0;
    }

    Log::log(LogLevel::INFO, "Dueling models", "BATCHER");

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
        StateResult result = env->getResult();

        switch(result)
        {
            case StateResult::BLACKWIN:
                if (is_swapped)
                    model2_wins++;
                else
                    model1_wins++;
                break;
            case StateResult::WHITEWIN:
                if (is_swapped)
                    model1_wins++;
                else
                    model2_wins++;
                break;
            case StateResult::DRAW:
                draws++;
                break;
            default:
                Log::log(LogLevel::ERROR, "Got non terminal result from env in duel models summary", "BATCHER");
                break;
        }
    }

    non_draws = model1_wins + model2_wins;
    draws = total_games - non_draws;

    win_delta = float(model2_wins - model1_wins) / non_draws;

    Log::log(LogLevel::INFO, "Duel models result:", "BATCHER");
    Log::log(LogLevel::INFO, "  Draws:       " + std::to_string(draws), "BATCHER");
    Log::log(LogLevel::INFO, "  Model1 wins: " + std::to_string(model1_wins) + " (" + models[0]->getName() + ")", "BATCHER");
    if (models[1] != nullptr)
        Log::log(LogLevel::INFO, "  Model2 wins: " + std::to_string(model2_wins) + " (" + models[1]->getName() + ")", "BATCHER");
    Log::log(LogLevel::INFO, "  Delta:       " + std::to_string(win_delta) + " (-1 is Model1 wins)", "BATCHER");


    return win_delta;
}

void Batcher::selfplay()
{
    Log::log(LogLevel::INFO, "Started selfplay", "BATCHER");
    // Run until terminal
    runGameloop();

    // Log outcome
    Log::log(LogLevel::INFO, "Selfplay result: " + std::to_string(averageWinner()) + " average winning color", "BATCHER");
}

void Batcher::humanplay(bool human_color)
{
    runNetwork();
    // TODO: Random bug with 1 sim and human winning
    std::cout << toString() << std::endl;

    while (!isTerminal())
    {
        if (environments[0]->getNextColor() == human_color)
        {
            index_t x, y;
            Utils::keyboardCordsInput(x, y);
            if (environments[0]->makeMove(x, y))
                runNetwork();
            else
                Log::log(LogLevel::ERROR, "Failed to perfom move!", "BATCHER");
        }
        else
        {
            runSimulations();
            if (Config::outputTrees())
            {
                Log::log(LogLevel::INFO, "Writing tree to: " + Config::outputTreesPath(), "BATCHER");
                Node* root = getEnvironment(0)->getCurrentNode();
                outputTree(root, -1);
                tree_viz_id++;
            }
            makeBestMoves();
        }

        renderEnvsHelper(true);
        freeMemory();
    }
}

Environment* Batcher::getEnvironment(int index)
{
    if (index < int(environments.size()))
        return environments[index];

    Log::log(LogLevel::FATAL, "Tried to get environment with out of bounds index", "BATCHER");
    return nullptr;
}

Node* Batcher::getNode(int index)
{
    if (index < int(environments.size()))
        return environments[index]->getCurrentNode();

    Log::log(LogLevel::FATAL, "Tried to get node with out of bounds index", "BATCHER");
    return nullptr;
}

void Batcher::freeMemory()
{
    Log::log(LogLevel::INFO, "Freeing memory", "BATCHER");

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

    if (amount == 0)
        return;

    if (mirrored)
    {
        if (environments.size() % 2 == 1)
        {
            Log::log(LogLevel::ERROR, "Tried to make mirrored random moves with uneven env count (" + std::to_string(environments.size()) + ")", "BATCHER");
            return;
        }

        Log::log(LogLevel::INFO, "Making " + std::to_string(amount) + " mirrored random moves", "BATCHER");

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
        Log::log(LogLevel::INFO, "Making " + std::to_string(amount) + " random moves", "BATCHER");

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
        Log::log(LogLevel::WARNING, "Got average winner before all environments finished playing", "BATCHER");
    }

    int total = environments.size() - non_terminal_environments.size();
    float delta = 0;

    for (Environment* env : environments)
    {
        // Skip if env is terminal
        if (!env->isTerminal())
            continue;

        StateResult result = env->getResult();
        switch(result)
        {
            case StateResult::BLACKWIN:
                delta -= 1;
                break;
            case StateResult::WHITEWIN:
                delta += 1;
                break;
            case StateResult::DRAW:
                break;
            default:
                Log::log(LogLevel::ERROR, "Got non terminal result from env in averageWinner", "BATCHER");
                break;
        }
    }

    delta = delta / total;

    Log::log(LogLevel::INFO, "Got average winner: " + std::to_string(delta) + " (Black is negative)", "BATCHER");

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

        if (max_envs > 1)
        {
            output << std::endl << std::endl << "        ᐊ";
            for (int i = 0; i < BoardSize; i++)
                output << "═";
            output << "╡ Environment: " << real_id << " ╞";
            for (int i = 0; i < BoardSize; i++)
                output << "═";
            output << "ᐅ" << std::endl << std::endl;
        }

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
        output << std::endl << std::endl;

        if (amount > 1)
        {
            std::cout << "        ᐊ";
            for (int i = 0; i < BoardSize; i++)
                output << "═";
            output << "╡ Environment: " << real_id << " ╞";
            for (int i = 0; i < BoardSize; i++)
                output << "═";
            output << "ᐅ" << std::endl << std::endl;
        }

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

void nodeCrawler(std::vector<Datapoint>& datapoints, Node* node, StateResult winner)
{
    // Ignore non fully explored nodes
    if (node->isFullyExpanded())
    {
        Datapoint data;
        data.moves = node->getMoveHistory();
        data.best_move = node->absBestChild()->getParentAction();
        // Change to reflect if current play won
        switch (winner)
        {
            case StateResult::BLACKWIN:
                if (node->getNextColor())
                    data.winner = 0;
                else
                    data.winner = 1;
                break;
            case StateResult::WHITEWIN:
                if (node->getNextColor())
                    data.winner = 1;
                else
                    data.winner = 0;
                break;
            case StateResult::DRAW:
                data.winner = 2;
                break;
            default:
                Log::log(LogLevel::ERROR, "Got non terminal result from env in duel models summary", "BATCHER");
                break;
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
        StateResult result = env->getResult();
        for (Node* root_node : env->getRootNodes())
        {
            nodeCrawler(datapoints, root_node, result);
        }
    }

    Storage interface = Storage(Path);
    Log::log(LogLevel::INFO, "Storing " + std::to_string(datapoints.size()) + " datapoints to: " + Path, "BATCHER");

    for (Datapoint data : datapoints)
        interface.storeDatapoint(data);
    interface.applyChanges();
}

void Batcher::outputTree(Node* root, int envid)
{
    std::string path = Config::outputTreesPath();
    path += std::to_string(tree_viz_id);
    if (envid != -1)
        path += "_" + std::to_string(envid);
    path += ".dot";
    std::ofstream out(path);
    TreeVisualizer::generateGraphvizCode(root, out);
    out.close();
}