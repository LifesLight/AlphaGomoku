#include "Config.h"
#include "Node.h"
#include "State.h"
#include "Model.h"
#include "Environment.h"


int main(int argc, const char* argv[])
{
    if (argc < 3)
    {
        std::cout << "[FATAL]: Missing arguments" << std::endl;
        return 0;
    }

    int simulations = 100;
    if (argc == 4)
        simulations = std::stoi(argv[3]);

    Environment::initialize();

    Model* nnb = new Model(argv[1], torch::kMPS, "Model 1");
    Model* nnw = new Model(argv[2], torch::kMPS, "Model 2");
    Environment* env = new Environment(nnb, nnw);

    int turn = 1;

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
            //Node* cn = env->getPlayedNode();
            //std::cout << Node::analytics(cn, {"VISITS", "POLICY", "MEAN", "VALUE"}) << std::endl;
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

        env->freeMemory();
    }
    
    Environment::deinitialize();
    return 1;
}
