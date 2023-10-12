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

    int simulations = 100;
    if (argc == 3)
        simulations = std::stoi(argv[2]);

    Environment::initialize();

    Model* neural_network = new Model(argv[1], torch::kMPS, "Testmodel");
    Environment* env = new Environment(neural_network, nullptr);

    int turn = 1;

    while (!env->isFinished())
    {
        if (turn % 2)
        {
            uint16_t computedMove = env->calculateNextMove(simulations);
            env->makeMove(computedMove);
            Node* cn = env->getNode(!env->getNextColor());
            std::cout << Environment::nodeAnalytics(cn, {"VISITS", "POLICY", "MEAN", "VALUE"}) << std::endl;
        }
        else
        {
            std::string x_string, y_string;
            uint8_t x, y;
            std::cout << "Move:" << std::endl << "X:";
            std::cin >> x_string;
            std::cout << "Y:";
            std::cin >> y_string;
            x = std::stoi(x_string);
            y = std::stoi(y_string);
            env->makeMove(x, y);
        }
        std::cout << env->toString() << std::endl;
        turn++;
    }
    
    return 1;
}
