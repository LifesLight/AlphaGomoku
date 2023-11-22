#include "Config.h"
#include "Model.h"
#include "Batcher.h"
#include "Style.h"
#include "Log.h"

// TODO: Move state to TempData and change node to gamestate to create only from parent pointers
// BATCHER stuck on deconstruction?!

int main(int argc, const char* argv[])
{
    // Try to configure logging
    if (Utils::getEnv("LOGGING") != "")
    {
        std::string log_level = Utils::getEnv("LOGGING");
        if (log_level == "info")
            Log::setLogLevel(LogLevel::INFO);
        else if (log_level == "warning")
            Log::setLogLevel(LogLevel::WARNING);
        else if (log_level == "error")
            Log::setLogLevel(LogLevel::ERROR);
        else if (log_level == "fatal")
            Log::setLogLevel(LogLevel::FATAL);
        else
        {
            Log::log(LogLevel::WARNING, "Invalid LOGGING environment variable, falling back to default");
            Log::setLogLevel(LogLevel::ERROR);
        }
    }

    std::map<std::string, std::string> args = Utils::parseArgv(argc, argv);

    // Check for help, will print out all possible arguments and refer to readme
    if (args.find("help") != args.end() || args.find("?") != args.end())
    {
        std::cout 
            << "Usage: ./AlphaGomoku [arguments]" << std::endl
            << "Arguments:" << std::endl
            << "  --help, -?                 Print this help message" << std::endl
            << "  --mode                     Mode to run the program in (duel, selfplay, human)" << std::endl
            << "  --model                    Name of the model to use (can be overwritten)" << std::endl
            << "  --simulations              Number of simulations to run per move" << std::endl
            << "  --environments             Number of environments to run in parallel" << std::endl
            << "  --randmoves                Number of random moves to make before starting" << std::endl
            << "  --humancolor               Color of the human player (0 = black, 1 = white)" << std::endl
            << "  --stone                    Stone skin to use for rendering" << std::endl
            << "  --board                    Board skin to use for rendering" << std::endl
            << "  --renderenvs               Render the environments" << std::endl
            << "  --renderanalytics          Render the analytics" << std::endl
            << "  --renderenvscount          Number of environments to render" << std::endl
            << "  --datapath                 Path to store the data" << std::endl
            << "  --modelpath                Path to store the models" << std::endl
            << "  --device                   Device to use for inference (cpu, cuda, mps)" << std::endl
            << "  --scalar                   Scalar to use for inference (float16, float32)" << std::endl
            << "  --threads                  Number of threads to use for inference" << std::endl
            << "  --batchsize                Batchsize cap for inference" << std::endl
            << "  --policybias               Policy bias to use for MCTS" << std::endl
            << "  --valuebias                Value bias to use for MCTS" << std::endl
            << "  --explorationbias          Exploration bias to use for MCTS" << std::endl
            << "  --model1                   Name of the first model to use" << std::endl
            << "  --model2                   Name of the second model to use" << std::endl;
        return 0;
    }

    // Required
    std::string mode, model1_name = "!!UNDEFINED!!", model2_name = "!!UNDEFINED!!";
    if (args.find("mode") != args.end())
    {
        mode = args["mode"];
        if (mode != "duel" && mode != "selfplay" && mode != "human")
        {
            ForcePrintln("[FATAL]: Invalid argument: mode needs to be duel, selfplay or human");
            return 1;
        }
    }
    else
    {
        ForcePrintln("[FATAL]: Missing arguments: mode");
        return 1;
    }

    if (args.find("model") != args.end())
    {
        model1_name = args["model"];
        model2_name = args["model"];
    }
    else if (args.find("model1") != args.end())
    {
        model1_name = args["model1"];
    }
    else
    {
        ForcePrintln("[FATAL]: Missing arguments: no model1");
        return 1;
    }

    // Optional
    // Check if overwrite for models is requested
    if (args.find("model1") != args.end())
        model1_name = args["model1"];
    if (args.find("model2") != args.end())
        model2_name = args["model2"];

    int simulations = DefaultSimulations, environments = DefaultEnvironments, randmoves = 0, humancolor = 0;

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

    try
    {
        if (args.find("batchsize") != args.end())
            Config::setMaxBatchsize(std::stoi(args["batchsize"]));
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: batchsize needs to be a number");
    }

    try
    {
        if (args.find("policybias") != args.end())
            Config::setPolicyBias(std::stof(args["policybias"]));
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: policybias needs to be a number");
    }

    try
    {
        if (args.find("valuebias") != args.end())
            Config::setValueBias(std::stof(args["valuebias"]));
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: valuebias needs to be a number");
    }

    try
    {
        if (args.find("explorationbias") != args.end())
            Config::setExplorationBias(std::stof(args["explorationbias"]));
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: explorationbias needs to be a number");
    }

    if (args.find("renderenvs") != args.end())
    {
        if (args["renderenvs"] == "true")
            Config::setRenderEnvs(true);
        else if (args["renderenvs"] == "false")
            Config::setRenderEnvs(false);
        else
            ForcePrintln("[FATAL]: Invalid argument: renderenvs needs to be a boolean");
    }

    if (args.find("renderanalytics") != args.end())
    {
        if (args["renderanalytics"] == "true")
            Config::setRenderAnalytics(true);
        else if (args["renderanalytics"] == "false")
            Config::setRenderAnalytics(false);
        else
            ForcePrintln("[FATAL]: Invalid argument: renderanalytics needs to be a boolean");
    }

    try
    {
        if (args.find("renderenvscount") != args.end())
            Config::setRenderEnvsCount(std::stoi(args["renderenvscount"]));
    }
    catch(const std::exception& e)
    {
        ForcePrintln("[FATAL]: Invalid argument: renderenvscount needs to be a number");
    }


    if (args.find("datapath") != args.end())
        Config::setDatapointPath(args["datapath"]);

    if (args.find("modelpath") != args.end())
        Config::setModelPath(args["modelpath"]);

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

    if (args.find("scalar") != args.end())
    {
        std::string scalar = args["scalar"];
        if (scalar == "float16" || scalar == "half")
            Config::setTorchScalar(torch::kFloat16);
        else if (scalar == "float32" || scalar == "float" || scalar == "full")
            Config::setTorchScalar(torch::kFloat32);
        else
        {
            ForcePrintln("[Warning]: Invalid argument: scalar needs to be float16 or float32 --> Falling back to default");
        }

        if (Config::torchInferenceDevice() == torch::kCPU && Config::torchScalar() == torch::kFloat16)
        {
            ForcePrintln("[Warning]: Invalid argument: scalar needs to be float32 when using cpu --> Falling back to default");
            Config::setTorchScalar(torch::kFloat32);
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