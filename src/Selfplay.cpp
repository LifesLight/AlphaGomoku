#include "Config.h"
#include "Node.h"
#include "State.h"
#include "Model.h"
#include "Environment.h"

// args:
// ModelBlack, ModelWhite, OutputPath, Simulations, Games
int main(int argc, const char* argv[])
{
    if (argc < 6)
    {
        std::cerr << "[FATAL]: Missing arguments" << std::endl;
        return 0;
    }

    // Load models
    Model* NN_black;
    Model* NN_white;
    try
    {
        NN_black = Utils::autoloadModel(argv[1], torch::kMPS);
        NN_white = Utils::autoloadModel(argv[2], torch::kMPS);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[FATAL]: Failed to load neural netorks" << std::endl << e.what() << '\n';
        return 0;
    }

    // Get parameters
    std::string output_path;
    int simulations, games;
    try
    {
        output_path = argv[3];
        simulations = std::stoi(argv[4]);
        games = std::stoi(argv[5]);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[FATAL]: Failed get parameters" << std::endl << e.what() << '\n';
        return 0;
    }

    return 1;
}