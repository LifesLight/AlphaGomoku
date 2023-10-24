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
    if (argc == 4)
        simulations = std::stoi(argv[3]);

    Model* nnb = Utils::autoloadModel(argv[1], torch::kMPS);
    Model* nnw = Utils::autoloadModel(argv[2], torch::kMPS);

    Batcher batcher(10, nnb, nnw);

    std::cout << "Success" << std::endl;


    //std::cout << batcher.getNode(0)->state->toString();
    for (int i = 0; i < 10; i++)
        std::cout << Node::analytics(batcher.getEnvironment(i)->getOpposingNode(), {"POLICY", "VALUE"});
    return 1;
}