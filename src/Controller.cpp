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
    Environment* env = new Environment(neural_network, neural_network);

    int turn = 0;

    while (!env->isFinished())
    {
        if (true)
        {
            uint16_t computedMove = env->calculateNextMove(simulations);
            bool success = env->makeMove(computedMove);
            if (!success)
            {
                std::cout << "Computer move failed" << std::endl;
                return 0;
            }
            Node* cn = env->getPlayedNode();
            std::cout << Node::analytics(cn, {"VISITS", "POLICY", "MEAN", "VALUE"}) << std::endl;
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
            bool success = env->makeMove(x, y);
            if (!success)
            {
                std::cout << "Human move failed" << std::endl;
                return 0;
            }
        }
        std::cout << env->toString() << std::endl;
        turn++;
    }
    
    Environment::deinitialize();
    return 1;
}
