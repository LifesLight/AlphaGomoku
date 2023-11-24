#include "Config.h"
#include "Model.h"
#include "Batcher.h"
#include "Style.h"
#include "Log.h"

// TODO: Move state to TempData and change node to gamestate to create only from parent pointers
// BATCHER stuck on deconstruction?!

std::vector<std::string> valid_args = {
    "help",
    "mode",
    "model",
    "model1",
    "model2",
    "simulations",
    "simulations1",
    "simulations2",
    "device",
    "device1",
    "device2",
    "scalar",
    "scalar1",
    "scalar2",
    "environments",
    "randmoves",
    "humancolor",
    "stones",
    "board",
    "renderenvs",
    "renderanalytics",
    "renderenvscount",
    "datapath",
    "modelpath",
    "threads",
    "batchsize",
    "policybias",
    "valuebias",
    "explorationbias",
    "gcptarget",
    "simtarget",
};

std::map<std::string, torch::Device> device_map = {
    {"cpu", torch::kCPU},
    {"cuda", torch::kCUDA},
    {"mps", torch::kMPS}
};

std::map<std::string, torch::ScalarType> scalar_map = {
    {"float16", torch::kFloat16},
    {"half", torch::kFloat16},
    {"float32", torch::kFloat32},
    {"float", torch::kFloat32},
    {"full", torch::kFloat32}
};

void printInfo()
{
    std::cout << "#### AlphaGomoku v." << Config::version() << " © Alexander Kurtz 2023 ####" << std::endl;
}

// Returns if help was called
bool checkForHelp(const std::map<std::string, std::string> args)
{
    if (args.find("help") != args.end() || args.find("?") != args.end())
    {
        std::cout << "Usage: ./AlphaGomoku [arguments]" << std::endl;
        // Print out all possible arguments
        std::cout << "Arguments:" << std::endl;
        for (std::string possible : valid_args)
        {
            std::cout << "--" << possible << std::endl;
        }
        return true;
    }
    return false;
}

void setupLogLevel()
{
    if (Utils::getEnv("LOGGING") != "NONE")
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
}

void applyConfigArgs(std::map<std::string, std::string>& args)
{
    try
    {
        if (args.find("simulations") != args.end())
            Config::setDefaultSimulations(std::stoi(args["simulations"]));
        if (args.find("device") != args.end())
        {
            auto it = device_map.find(args["device"]);
            if (it != device_map.end())
            {
                Config::setTorchInferenceDevice(it->second);
            }
            else
            {
                Log::log(LogLevel::WARNING, "Invalid argument: device needs to be cpu, cuda or mps");
            }
        }
        if (args.find("scalar") != args.end())
        {
            if (scalar_map.find(args["scalar"]) != scalar_map.end())
                Config::setTorchScalar(scalar_map[args["scalar"]]);
            else
                Log::log(LogLevel::WARNING, "Invalid argument: scalar needs to be float16 or float32");
        }
        if (args.find("randmoves") != args.end())
            Config::setRandMoves(std::stoi(args["randmoves"]));
        if (args.find("humancolor") != args.end())
        {
            if (args["humancolor"] == "black" || args["humancolor"] == "0")
                Config::setHumanColor(0);
            else if (args["humancolor"] == "white" || args["humancolor"] == "1")
                Config::setHumanColor(1);
            else
                Log::log(LogLevel::WARNING, "Invalid argument: humancolor needs to be black or white");
        }
        if (args.find("environments") != args.end())
            Config::setEnvironmentCount(std::stoi(args["environments"]));
        if (args.find("datapath") != args.end())
            Config::setDatapointPath(args["datapath"]);
        if (args.find("modelpath") != args.end())
            Config::setModelPath(args["modelpath"]);
        if (args.find("gcptarget") != args.end())
            Config::setGamestatesPerThread(std::stoi(args["gcptarget"]));
        if (args.find("simtarget") != args.end())
            Config::setSimsPerThread(std::stoi(args["simtarget"]));
        if (args.find("policybias") != args.end())
            Config::setPolicyBias(std::stof(args["policybias"]));
        if (args.find("valuebias") != args.end())
            Config::setValueBias(std::stof(args["valuebias"]));
        if (args.find("explorationbias") != args.end())
            Config::setExplorationBias(std::stof(args["explorationbias"]));
        if (args.find("threads") != args.end())
            Config::setMaxThreads(std::stoi(args["threads"]));
        if (args.find("batchsize") != args.end())
            Config::setMaxBatchsize(std::stoi(args["batchsize"]));
        if (args.find("renderenvs") != args.end())
        {
            if (args["renderenvs"] == "true")
                Config::setRenderEnvs(true);
            else if (args["renderenvs"] == "false")
                Config::setRenderEnvs(false);
            else
                Log::log(LogLevel::WARNING, "Invalid argument: renderenvs needs to be a boolean");
        }
        if (args.find("renderanalytics") != args.end())
        {
            if (args["renderanalytics"] == "true")
                Config::setRenderAnalytics(true);
            else if (args["renderanalytics"] == "false")
                Config::setRenderAnalytics(false);
            else
                Log::log(LogLevel::WARNING, "Invalid argument: renderanalytics needs to be a boolean");
        }
        if (args.find("renderenvscount") != args.end())
            Config::setRenderEnvsCount(std::stoi(args["renderenvscount"]));
    }
    catch(const std::exception& e)
    {
        Log::log(LogLevel::WARNING, "Invalid argument format: " + std::string(e.what()));
    }
}

void applyStyleArgs(std::map<std::string, std::string>& args)
{
    if (args.find("stones") != args.end())
        Style::setStone(args["stones"]);

    if (args.find("board") != args.end())
        Style::setBoard(args["board"]);
}

void warnInvalidArgs(std::map<std::string, std::string>& args)
{
    for (auto const& [key, value] : args)
    {
        if (std::find(valid_args.begin(), valid_args.end(), key) == valid_args.end())
            Log::log(LogLevel::WARNING, "Invalid argument: " + key);
    }
}

std::tuple<Model*, Model*> configModels(std::map<std::string, std::string>& args)
{
    std::string model1_name = "!!UNDEFINED!!";
    std::string model2_name = "!!UNDEFINED!!";

    if (args.find("model") != args.end())
    {
        model1_name = args["model"];
        model2_name = args["model"];
    }
    else if (args.find("model1") != args.end())
    {
        model1_name = args["model1"];
    }

    if (args.find("model2") != args.end())
    {
        model2_name = args["model2"];
    }

    Model* model_1 = nullptr;
    Model* model_2 = nullptr;

    if (model1_name != "!!UNDEFINED!!")
        model_1 = Model::autoloadModel(model1_name);

    if (model2_name != "!!UNDEFINED!!")
        model_2 = Model::autoloadModel(model2_name);

    // Configure Models
    if (model_1 != nullptr)
    {
        try
        {
            if (args.find("simulations1") != args.end())
                model_1->setSimulations(std::stoi(args["simulations1"]));
            if (args.find("device1") != args.end())
            {
                auto it = device_map.find(args["device1"]);
                if (it != device_map.end())
                {
                    model_1->setDevice(it->second);
                }
                else
                {
                    Log::log(LogLevel::WARNING, "Invalid argument in model1 config: device1 needs to be cpu, cuda or mps");
                }
            }
            if (args.find("scalar1") != args.end())
            {
                if (scalar_map.find(args["scalar1"]) != scalar_map.end())
                    model_1->setPrec(scalar_map[args["scalar1"]]);
                else
                    Log::log(LogLevel::WARNING, "Invalid argument in model1 config: scalar1 needs to be float16 or float32");
            }
        }
        catch(const std::exception& e)
        {
            Log::log(LogLevel::WARNING, "Invalid argument in model1 config: " + std::string(e.what()));
        }
    }

    if (model_2 != nullptr)
    {
        try
        {
            if (args.find("simulations2") != args.end())
                model_2->setSimulations(std::stoi(args["simulations2"]));
            if (args.find("device2") != args.end())
            {
                auto it = device_map.find(args["device2"]);
                if (it != device_map.end())
                {
                    model_2->setDevice(it->second);
                }
                else
                {
                    Log::log(LogLevel::WARNING, "Invalid argument in model2 config: device2 needs to be cpu, cuda or mps");
                }
            }
            if (args.find("scalar2") != args.end())
            {
                if (scalar_map.find(args["scalar2"]) != scalar_map.end())
                    model_2->setPrec(scalar_map[args["scalar2"]]);
                else
                    Log::log(LogLevel::WARNING, "Invalid argument in model2 config: scalar2 needs to be float16 or float32");
            }
        }
        catch(const std::exception& e)
        {
            Log::log(LogLevel::WARNING, "Invalid argument in model2 config: " + std::string(e.what()));
        }
    }

    return std::make_tuple(model_1, model_2);
}

bool runDuel(Model* model_1, Model* model_2)
{
    if (model_1 == nullptr || model_2 == nullptr)
    {
        Log::log(LogLevel::FATAL, "Missing argument: duel needs two models");
        return 1;
    }

    Batcher* batcher = new Batcher(Config::environmentCount(), model_1, model_2);
    batcher->swapModels();
    if (Config::randMoves() > 0)
        batcher->makeRandomMoves(Config::randMoves(), true);
    batcher->duelModels();

    delete model_1;
    delete model_2;
    delete batcher;
    return 0;
}

bool runSelfplay(Model* model_1)
{
    if (model_1 == nullptr)
    {
        Log::log(LogLevel::FATAL, "Missing argument: selfplay needs a model");
        return 1;
    }
    Batcher* batcher = new Batcher(Config::environmentCount(), model_1);
    if (Config::randMoves() > 0)
        batcher->makeRandomMoves(Config::randMoves(), false);
    batcher->selfplay();
    batcher->storeData(Config::datapointPath());

    delete model_1;
    delete batcher;
    return 0;
}

bool runHumanplay(Model* model_1)
{
    if (model_1 == nullptr)
    {
        Log::log(LogLevel::FATAL, "Missing argument: humanplay needs a model");
        return 1;
    }
    Batcher* batcher = new Batcher(1, model_1);
    if (Config::randMoves() > 0)
        batcher->makeRandomMoves(Config::randMoves(), false);
    batcher->humanplay(Config::humanColor());

    delete model_1;
    delete batcher;
    return 0;
}

bool runMode(std::string mode, Model* model_1, Model* model_2)
{
    if (mode == "duel")
        return runDuel(model_1, model_2);
    else if (mode == "selfplay")
        return runSelfplay(model_1);
    else if (mode == "human")
        return runHumanplay(model_1);
    else
    {
        Log::log(LogLevel::FATAL, "Invalid argument: mode needs to be duel, selfplay or human");
        return 1;
    }
}

int main(int argc, const char* argv[])
{
    Config::setVersion("0.1.0");
    printInfo();
    setupLogLevel();

    std::map<std::string, std::string> args = Utils::parseArgv(argc, argv);
    if (checkForHelp(args))
        return 0;

    applyConfigArgs(args);
    applyStyleArgs(args);
    warnInvalidArgs(args);

    Model* model_1 = nullptr;
    Model* model_2 = nullptr;
    std::tuple<Model*, Model*> models = configModels(args);
    model_1 = std::get<0>(models);
    model_2 = std::get<1>(models);

    if (args.find("mode") == std::end(args))
    {
        Log::log(LogLevel::FATAL, "Missing argument: mode");
        return 1;
    }

    return runMode(args["mode"], model_1, model_2);
}