#pragma once

#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <vector>
#include <list>
#include <random>
#include <algorithm>
#include <string>
#include <chrono>
#include <cstring>
#include <tuple>
#include <functional>
#include <filesystem>
#include <sstream>
#include <initializer_list>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

/* These are the default values, can easily be overwritten by calling for changes in Config */

// External data paths
#define ModelPath "../Models/scripted/"
#define DatapointPath "../Datasets/Selfplay/data.txt"

// At least 2 and even number
#define HistoryDepth 8

// How many datapoints should be stored in a selfplay dataset
#define MaxDatapoints 50'000

// MCTS Master parameters
#define DefaultSimulations 1600
#define DefaultEnvironments 10

// Algorithm Hyperparameters
#define ExplorationBias 1
#define PolicyBias 1
#define ValueBias 2


// ---- Performance Settings ----
// This is max threads PER task, so could be MaxThreads * 2 effective threads
// To disable threading just set to 1 --> will use main thread
#define MaxThreads 4
// These are target values, will not always be matched
// How many simulations a thread should aim to handle
#define PerThreadSimulations 64
// How many nodes a single thread should convert to gamestates
#define PerThreadGamestateConvertions 32

// Torch Settings
// This is where tensors are created and simmelar
#define TorchDefaultDevice torch::kCPU
// This is the device computations will be run on
#define TorchInferenceDevice torch::kCPU
// Floating point precision for Inference
#define TorchDefaultScalar torch::kFloat32
// Higher is better if VRAM/RAM can handle
#define MaxBatchsize 2048
// -------------------------------

#define RenderEnvs true
#define RenderAnalytics false
#define RenderEnvsCount 1

/* These need to be pre compiler definitions, will need to recompile to change */

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

// Max children per node, 0 is no limit
#define BranchingLimit 0

// Save memory if 2d -> 1d index mapping fits in 2^8
#if Boardsize < 16
typedef uint8_t index_t;
#else
typedef uint16_t index_t;
#endif


class Config
{
private:
    static std::string model_path;
    static std::string datapoint_path;
    static std::string build_version;
    static int history_depth;
    static int max_datapoints;
    static int default_simulations;
    static int default_environments;
    static float exploration_bias;
    static float policy_bias;
    static float value_bias;
    static int max_threads;
    static int sims_per_thread;
    static int gamestates_per_thread;
    static torch::Device torch_host_device;
    static torch::Device torch_inference_device;
    static torch::ScalarType torch_scalar;
    static int max_batchsize;
    static bool render_envs;
    static bool render_analytics;
    static int render_envs_count;

public:
    static std::string modelPath();
    static std::string datapointPath();
    static int historyDepth();
    static int maxDatapoints();
    static int defaultSimulations();
    static int defaultEnvironments();
    static float explorationBias();
    static float policyBias();
    static float valueBias();
    static int maxThreads();
    static int simsPerThread();
    static int gamestatesPerThread();
    static torch::Device torchHostDevice();
    static torch::Device torchInferenceDevice();
    static torch::ScalarType torchScalar();
    static int maxBatchsize();
    static bool renderEnvs();
    static bool renderAnalytics();
    static int renderEnvsCount();
    static std::string version();

    static void setModelPath(std::string path);
    static void setDatapointPath(std::string path);
    static void setHistoryDepth(int depth);
    static void setMaxDatapoints(int datapoints);
    static void setDefaultSimulations(int sims);
    static void setDefaultEnvironments(int envs);
    static void setExplorationBias(float bias);
    static void setPolicyBias(float bias);
    static void setValueBias(float bias);
    static void setMaxThreads(int threads);
    static void setSimsPerThread(int sims);
    static void setGamestatesPerThread(int gamestates);
    static void setTorchHostDevice(torch::Device device);
    static void setTorchInferenceDevice(torch::Device device);
    static void setTorchScalar(torch::ScalarType scalar);
    static void setMaxBatchsize(int batchsize);
    static void setRenderEnvs(bool render);
    static void setRenderAnalytics(bool render);
    static void setRenderEnvsCount(int count);
    static void setVersion(std::string version);
};