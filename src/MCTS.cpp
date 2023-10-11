#include "Config.h"
#include "Node.h"
#include "State.h"
#include "Model.h"
#include "Environment.h"


int main(int argc, const char* argv[])
{
    if (argc == 1)
    {
        std::cout << "[FATAL]: Missing model path" << std::endl;
        return 0;
    }

    Model* neural_network = new Model(argv[1], torch::kMPS, "Testmodel");

    Environment* env = new Environment(neural_network, neural_network);

    while (!env->isFinished())
    {
        uint16_t computedMove = env->calculateNextMove(500);
        env->makeMove(computedMove);
        std::cout << env->toString() << std::endl;
    }
    
    return 1;
}
