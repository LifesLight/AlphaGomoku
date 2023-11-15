#include "Config.h"
#include "Model.h"
#include "Batcher.h"

void superSelfplayWorker(Batcher* batcher)
{
    batcher->selfplay();
}

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        std::cout << "[FATAL]: Missing arguments" << std::endl;
        std::cout << "Usage: [DUEL, SELFPLAY, HUMAN] ..." << std::endl;
        return 0;
    }

    // Selected game mode
    std::string mode = argv[1];

    // Duel is evaluate 2 models against each other
    if (mode == "DUEL")
    {
        if (argc < 4)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: DUEL Model1 Model2 [Simulations] [Environments] [Rand Moves Count]" << std::endl;
            return 0;
        }

        Model* model_1 = Model::autoloadModel(argv[2]);
        Model* model_2 = Model::autoloadModel(argv[3]);

        // Get hyperparameters
        int simulations = DefaultSimulations;
        int environment_count = DefaultEnvironments;
        int rand_moves = 0;

        if (argc > 4)
            simulations = std::stoi(argv[4]);

        if (argc > 5)
            environment_count = std::stoi(argv[5]);

        if (argc > 6)
            rand_moves = std::stoi(argv[6]);

        model_1->setSimulations(1);
        model_2->setSimulations(simulations);

        Batcher* batcher = new Batcher(environment_count, model_1, model_2);
        batcher->swapModels();
        batcher->makeRandomMoves(rand_moves, true);
        batcher->duelModels();
        ForcePrintln("This was " << ExplorationBias);
        delete batcher;
    }

    // Selfplay
    if (mode == "SELFPLAY")
    {
        if (argc < 3)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: SELFPLAY Model [Simulations] [Environments] [Rand Moves Count]" << std::endl;
            return 0;
        }

        Model* model_1 = Model::autoloadModel(argv[2]);

        // Get hyperparameters
        int simulations = DefaultSimulations;
        int environment_count = DefaultEnvironments;
        int rand_moves = 0;

        if (argc > 3)
            simulations = std::stoi(argv[3]);

        if (argc > 4)
            environment_count = std::stoi(argv[4]);

        if (argc > 5)
            rand_moves = std::stoi(argv[5]);

        model_1->setSimulations(simulations);

        Batcher* batcher = new Batcher(environment_count, model_1);
        batcher->makeRandomMoves(rand_moves, false);
        batcher->selfplay();
        batcher->storeData(DatapointPath);
        delete batcher;
    }

    // Super Selfplay is basically selfplay but highly multithreaded with multiple batchers working in parallel
    if (mode == "SUPER_SELFPLAY")
    {
        std::cout << "[Controller]: !!WARNING!! Super Selfplay is in experimental stage and only intented for high end hardware" << std::endl;
        if (argc < 4)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: SUPER_SELFPLAY Model Threading Factor [Simulations] [Environments] [Rand Moves Count]" << std::endl;
            return 0;
        }

        Model* model_1 = Model::autoloadModel(argv[2]);

        // Get hyperparameters
        int threading_factor = 2;
        int simulations = DefaultSimulations;
        int environment_count = DefaultEnvironments;
        int rand_moves = 0;

        if (argc > 3)
            threading_factor = std::stoi(argv[3]);

        if (argc > 4)
            simulations = std::stoi(argv[4]);

        if (argc > 5)
            environment_count = std::stoi(argv[5]);

        if (argc > 6)
            rand_moves = std::stoi(argv[6]);

        std::vector<Batcher*> batchers;
        std::vector<std::thread> threads;
        batchers.reserve(threading_factor);
        threads.reserve(threading_factor);

        model_1->setSimulations(simulations);

        for (int i = 0; i < threading_factor; i++)
        {
            Batcher* batcher = new Batcher(environment_count, model_1);
            batcher->makeRandomMoves(rand_moves, false);
            batchers.push_back(batcher);
            threads.emplace_back(superSelfplayWorker, batcher);
        }

        for (int i = 0; i < threading_factor; i++)
            threads[i].join();

        for (Batcher* batcher : batchers)
        {
            batcher->storeData(DatapointPath);
            delete batcher;
        }
    }

    // Human play
    if (mode == "HUMAN")
    {
        if (argc < 3)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: SELFPLAY Model [Simulations] [Human Color]" << std::endl;
            return 0;
        }

        Model* model_1 = Model::autoloadModel(argv[2]);

        int simulations = DefaultSimulations;
        int human_color = 0;

        if (argc > 3)
            simulations = std::stoi(argv[3]);

        if (argc > 4)
            human_color = std::stoi(argv[4]);

        model_1->setSimulations(simulations);

        Batcher* batcher = new Batcher(1, model_1);
        batcher->humanplay(human_color);
        delete batcher;
    }

    return 0;
}