#include "Config.h"
#include "Model.h"
#include "Batcher.h"
#include "Style.h"

// TODO: Move state to TempData and change node to gamestate to create only from parent pointers
// BATCHER stuck on deconstruction?!

int main(int argc, const char* argv[])
{
    std::map<std::string, std::string> args = Utils::parseArgv(argc, argv);

    // Required
    std::string mode, model1_name;
    if (args.find("mode") != args.end())
        mode = args["mode"];
    else
    {
        ForcePrintln("[FATAL]: Missing arguments: mode");
        return 1;
    }

    if (args.find("model1") != args.end())
        model1_name = args["model1"];
    else
    {
        ForcePrintln("[FATAL]: Missing arguments: model1");
        return 1;
    }

    // Optional
    int simulations = DefaultSimulations, environments = DefaultEnvironments, randmoves = 0, humancolor = 0;
    std::string model2_name = "!!UNDEFINED!!";
    if (args.find("model2") != args.end())
        model2_name = args["model2"];

    if (args.find("stone") != args.end())
        Style::setStone(args["stone"]);

    if (args.find("board") != args.end())
        Style::setBoard(args["board"]);

    try 
    {
        if (args.find("simulations") != args.end())
            simulations = std::stoi(args["simulations"]);
    }
    catch (const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: simulations needs to be a number");
        return 1;
    }

    try
    {
        if (args.find("environments") != args.end())
            environments = std::stoi(args["environments"]);
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: environments needs to be a number");
        return 1;
    }

    try
    {
        if (args.find("randmoves") != args.end())
            randmoves = std::stoi(args["randmoves"]);
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: randmoves needs to be a number");
        return 1;
    }

    try
    {
        if (args.find("humancolor") != args.end())
            humancolor = std::stoi(args["humancolor"]);
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: humancolor needs to be a number");
        return 1;
    }

    try
    {
        if (args.find("threads") != args.end())
            Config::setMaxThreads(std::stoi(args["threads"]));
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    if (args.find("datapath") != args.end())
        Config::setDatapointPath(args["datapath"]);

    if (args.find("modelpath") != args.end())
        Config::setModelPath(args["modelpath"]);

    if (args.find("scalar") != args.end())
    {
        std::string scalar = args["scalar"];
        if (scalar == "float16")
            Config::setTorchScalar(torch::kFloat16);
        else if (scalar == "float32")
            Config::setTorchScalar(torch::kFloat32);
        else
        {
            ForcePrintln("[Warning]: Invalid argument: scalar needs to be float16 or float32 --> Falling back to default");
        }
    }

    if (args.find("device") != args.end())
    {
        std::string device = args["device"];
        if (device == "cpu")
            Config::setTorchInferenceDevice(torch::kCPU);
        else if (device == "cuda")
            Config::setTorchInferenceDevice(torch::kCUDA);
        else if (device == "mps")
            Config::setTorchInferenceDevice(torch::kMPS);
        else
        {
            ForcePrintln("[Warning]: Invalid argument: device needs to be cpu, cuda or mps --> Falling back to default");
        }
    }

    // Duel is evaluate 2 models against each other
    if (mode == "duel")
    {
        if (model2_name == "!!UNDEFINED!!")
        {
            ForcePrintln("[FATAL]: Missing arguments for duel: model2");
            return 1;
        }

        Model* model_1 = Model::autoloadModel(model1_name);
        Model* model_2 = Model::autoloadModel(model2_name);

        model_1->setSimulations(simulations);
        model_2->setSimulations(simulations);

        Batcher* batcher = new Batcher(environments, model_1, model_2);
        batcher->swapModels();
        batcher->makeRandomMoves(randmoves, true);
        batcher->duelModels();

        delete model_1;
        delete model_2;
        delete batcher;
    }

    // Selfplay
    if (mode == "selfplay")
    {
        Model* model_1 = Model::autoloadModel(model1_name);
        model_1->setSimulations(simulations);

        Batcher* batcher = new Batcher(environments, model_1);
        batcher->makeRandomMoves(randmoves, false);
        batcher->selfplay();
        batcher->storeData(Config::datapointPath());

        delete model_1;
        delete batcher;
    }

    // Human play
    if (mode == "human")
    {
        Model* model_1 = Model::autoloadModel(model1_name);

        model_1->setSimulations(simulations);

        Batcher* batcher = new Batcher(1, model_1);
        batcher->humanplay(humancolor);

        delete model_1;
        delete batcher;
    }

    return 0;
}