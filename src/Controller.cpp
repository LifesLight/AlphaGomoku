#include "Config.h"
#include "Node.h"
#include "State.h"
#include "Model.h"
#include "Environment.h"
#include "Batcher.h"

int main(int argc, const char* argv[])
{
    if (argc < 3)
    {
        std::cout << "[FATAL]: Missing arguments" << std::endl;
        return 0;
    }

    int simulations = BoardSize * BoardSize;
    int environment_count = 10;
    int rand_moves = 0;
    if (argc >= 4)
        simulations = std::stoi(argv[3]);

    if (argc >= 5)
        environment_count = std::stoi(argv[4]);

    if (argc >= 6)
        rand_moves = std::stoi(argv[5]);

    Model* nnb = Model::autoloadModel(argv[1]);
    Model* nnw = Model::autoloadModel(argv[2]);

    Batcher batcher(environment_count, nnb);
    batcher.swapModels();
    std::cout << batcher.duelModels(rand_moves, simulations) << std::endl;

    //batcher.storeData(argv[2]);
    /*
    batcher.makeRandomMoves(rand_moves);
    while (!batcher.isTerminal())
    {
        batcher.runSimulations(simulations);
        batcher.makeBestMoves();
        std::cout << batcher.toStringDist({"MEAN", "VALUE", "VISITS", "POLICY"}) << std::endl;
    }

    //batcher.storeData("../Datasets/Selfplay/Datapoints");
    std::cout << batcher.averageWinner() << std::endl;
    */
    batcher.freeMemory();

    return 1;
}