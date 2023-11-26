#include "Config.h"

// Default values
std::string Config::model_path = ModelPath;
std::string Config::datapoint_path = DatapointPath;
int Config::history_depth = HistoryDepth;
int Config::max_datapoints = MaxDatapoints;
int Config::default_simulations = DefaultSimulations;
float Config::exploration_bias = ExplorationBias;
float Config::policy_bias = PolicyBias;
float Config::value_bias = ValueBias;
int Config::max_threads = MaxThreads;
int Config::sims_per_thread = PerThreadSimulations;
int Config::gamestates_per_thread = PerThreadGamestateConvertions;
torch::Device Config::torch_host_device = TorchDefaultDevice;
torch::Device Config::torch_inference_device = TorchInferenceDevice;
torch::ScalarType Config::torch_scalar = TorchDefaultScalar;
int Config::max_batchsize = MaxBatchsize;
bool Config::render_envs = true;
bool Config::render_analytics = false;
int Config::render_envs_count = 1;
int Config::randmoves = 0;
bool Config::human_color = 0;
int Config::environment_count = 10;
std::string Config::build_version = "unspecified";
bool Config::output_trees = false;
std::string Config::output_trees_path = TreesPath;
bool Config::nocache = false;
int Config::rng_seed = -1;

std::string Config::version()
{
    return build_version;
}

std::string Config::modelPath()
{
    return model_path;
}

std::string Config::datapointPath()
{
    return datapoint_path;
}

int Config::historyDepth()
{
    return history_depth;
}

int Config::maxDatapoints()
{
    return max_datapoints;
}

int Config::defaultSimulations()
{
    return default_simulations;
}

int Config::environmentCount()
{
    return environment_count;
}

float Config::explorationBias()
{
    return exploration_bias;
}

float Config::policyBias()
{
    return policy_bias;
}

float Config::valueBias()
{
    return value_bias;
}

int Config::maxThreads()
{
    return max_threads;
}

int Config::simsPerThread()
{
    return sims_per_thread;
}

int Config::gamestatesPerThread()
{
    return gamestates_per_thread;
}

torch::Device Config::torchHostDevice()
{
    return torch_host_device;
}

torch::Device Config::torchInferenceDevice()
{
    return torch_inference_device;
}

torch::ScalarType Config::torchScalar()
{
    return torch_scalar;
}

int Config::maxBatchsize()
{
    return max_batchsize;
}

bool Config::renderEnvs()
{
    return render_envs;
}

bool Config::renderAnalytics()
{
    return render_analytics;
}

int Config::renderEnvsCount()
{
    return render_envs_count;
}

int Config::randMoves()
{
    return randmoves;
}

bool Config::humanColor()
{
    return human_color;
}

bool Config::outputTrees()
{
    return output_trees;
}

std::string Config::outputTreesPath()
{
    return output_trees_path;
}

bool Config::noCache()
{
    return nocache;
}

int Config::seed()
{
    return rng_seed;
}

void Config::setModelPath(std::string path)
{
    model_path = path;
}

void Config::setDatapointPath(std::string path)
{
    datapoint_path = path;
}

void Config::setHistoryDepth(int depth)
{
    history_depth = depth;
}

void Config::setMaxDatapoints(int datapoints)
{
    max_datapoints = datapoints;
}

void Config::setDefaultSimulations(int sims)
{
    default_simulations = sims;
}

void Config::setEnvironmentCount(int envs)
{
    environment_count = envs;
}

void Config::setExplorationBias(float bias)
{
    exploration_bias = bias;
}

void Config::setPolicyBias(float bias)
{
    policy_bias = bias;
}

void Config::setValueBias(float bias)
{
    value_bias = bias;
}

void Config::setMaxThreads(int threads)
{
    max_threads = threads;
}

void Config::setSimsPerThread(int sims)
{
    sims_per_thread = sims;
}

void Config::setGamestatesPerThread(int gamestates)
{
    gamestates_per_thread = gamestates;
}

void Config::setTorchHostDevice(torch::Device device)
{
    torch_host_device = device;
}

void Config::setTorchInferenceDevice(torch::Device device)
{
    torch_inference_device = device;
}

void Config::setTorchScalar(torch::ScalarType scalar)
{
    torch_scalar = scalar;
}

void Config::setMaxBatchsize(int batchsize)
{
    max_batchsize = batchsize;
}

void Config::setRenderEnvs(bool render)
{
    render_envs = render;
}

void Config::setRenderAnalytics(bool render)
{
    render_analytics = render;
}

void Config::setRenderEnvsCount(int count)
{
    render_envs_count = count;
}

void Config::setVersion(std::string version)
{
    Config::build_version = version;
}

void Config::setRandMoves(int moves)
{
    randmoves = moves;
}

void Config::setHumanColor(bool color)
{
    human_color = color;
}

void Config::setOutputTrees(bool output)
{
    output_trees = output;
}

void Config::setOutputTreesPath(std::string path)
{
    output_trees_path = path;
}

void Config::setNoCache(bool cache)
{
    nocache = cache;
}

void Config::setSeed(int seed)
{
    rng_seed = seed;
}