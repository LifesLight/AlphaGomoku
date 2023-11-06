#include "Config.h"
#include "Model.h"
#include "Batcher.h"

int main(int argc, const char* argv[])
{
    if (argc < 2)
    {
        std::cout << "[FATAL]: Missing arguments" << std::endl;
        std::cout << "Usage: [Model1 Path] (Model2 Path) (Simulations) (Environments) (Random Moves) (Duel 1 / 0)" << std::endl;
        return 0;
    }

    // Determine Tree mode
    bool single_tree = 1;
    if (argc > 2)
        if (!std::isdigit(argv[2][0]))
            single_tree = 0;

    // Get hyperparameters
    int simulations = DefaultSimulations;
    int environment_count = DefaultEnvironments;
    int rand_moves = 0;
    bool duel = 0;

    if (argc >= 4 - single_tree)
        simulations = std::stoi(argv[3 - single_tree]);

    if (argc >= 5 - single_tree)
        environment_count = std::stoi(argv[4 - single_tree]);

    if (argc >= 6 - single_tree)
        rand_moves = std::stoi(argv[5 - single_tree]);

    if (argc >= 7 - single_tree)
        duel = std::stoi(argv[6 - single_tree]);

    // Load models
    Batcher* batcher = nullptr;
    if (single_tree)
    {
        Model* nn = Model::autoloadModel(argv[1]);
        batcher = new Batcher(environment_count, nn);
    }
    else
    {
        Model* nnb = Model::autoloadModel(argv[1], simulations);
        Model* nnw = Model::autoloadModel(argv[2], simulations);
        batcher = new Batcher(environment_count, nnb, nnw);
    }

    // Either duel or selfplay
    if (duel)
    {
        batcher->swapModels();
        float win_delta = batcher->duelModels(rand_moves, simulations);
        std::cout << "Win delta: " << win_delta << std::endl;
    }
    else
    {
        batcher->makeRandomMoves(rand_moves);
        batcher->selfplay(simulations);
        float average_winner = batcher->averageWinner();
        std::cout << "Average Winner: " << average_winner << std::endl;
        batcher->storeData(DatapointPath);
    }

    delete batcher;
    return 0;
}