#include "Config.h"
#include "Node.h"
#include "State.h"
#include "Model.h"
#include "Environment.h"


int main(int argc, const char* argv[])
{
    Model* neural_network = new Model(argv[1], torch::kCUDA, "Testmodel");

    Environment* env = new Environment(neural_network, neural_network);

    while (!env->isFinished())
    {
        uint16_t computedMove = env->calculateNextMove(500);
        env->makeMove(computedMove);
        std::cout << env->toString() << std::endl;
    }
}
