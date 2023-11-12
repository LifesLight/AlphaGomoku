#include "Config.h"
#include "Model.h"
#include "Batcher.h"

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
    Batcher* batcher;

    // Duel is evaluate 2 models against each other
    if (mode == "DUEL")
    {
        if (argc < 4)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: DUEL [Model1] [Model2] [Simulations] [Environments] [Rand Moves Count]" << std::endl;
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

        model_1->setSimulations(800);
        model_2->setSimulations(simulations);

        batcher = new Batcher(environment_count, model_1, model_2);
        batcher->swapModels();
        batcher->makeRandomMoves(rand_moves, true);
        batcher->duelModels();
    }

    // Selfplay
    if (mode == "SELFPLAY")
    {
        if (argc < 3)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: SELFPLAY [Model] [Simulations] [Environments] [Rand Moves Count]" << std::endl;
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

        batcher = new Batcher(environment_count, model_1);
        batcher->makeRandomMoves(rand_moves, false);
        batcher->selfplay();
        batcher->storeData(DatapointPath);
    }

    // Human play
    if (mode == "HUMAN")
    {
        if (argc < 3)
        {
            std::cout << "[FATAL]: Missing arguments" << std::endl;
            std::cout << "Usage: SELFPLAY [Model] [Simulations] [Human Color]" << std::endl;
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

        batcher = new Batcher(1, model_1);
        batcher->humanplay(human_color);
    }

    delete batcher;
    return 0;
}