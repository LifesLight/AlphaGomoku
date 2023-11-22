#include "Config.h"

std::string Config::model_path = ModelPath;
std::string Config::datapoint_path = DatapointPath;
int Config::history_depth = HistoryDepth;
int Config::max_datapoints = MaxDatapoints;
int Config::default_simulations = DefaultSimulations;
int Config::default_environments = DefaultEnvironments;
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
bool Config::render_envs = RenderEnvs;
bool Config::render_analytics = RenderAnalytics;
int Config::render_envs_count = RenderEnvsCount;

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

int Config::defaultEnvironments()
{
    return default_environments;
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

void Config::setDefaultEnvironments(int envs)
{
    default_environments = envs;
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