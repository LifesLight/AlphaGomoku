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
    if (argc >= 4)
        simulations = std::stoi(argv[3]);

    if (argc >= 5)
        environment_count = std::stoi(argv[4]);

    Model* nnb = Utils::autoloadModel(argv[1], torch::kCPU);
    Model* nnw = Utils::autoloadModel(argv[2], torch::kCPU);

    Batcher batcher(environment_count, nnb, nnw);

    batcher.makeRandomMoves(5);
    batcher.runSimulations(simulations);
    batcher.makeBestMoves();

    for (int i = 0; i < environment_count; i++)
    {
        std::cout << Node::analytics(batcher.getEnvironment(i)->getCurrentNode(), {"POLICY", "VALUE", "MEAN", "VISITS"});
    }

    batcher.freeMemory();

    return 1;
}