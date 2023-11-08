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

// External data paths
#define ModelPath "../Models/scripted/"
#define DatapointPath "../Datasets/Selfplay/data.txt"

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

// At least 2 and even number
#define HistoryDepth 8

// How many datapoints should be stored in a selfplay dataset
#define MaxDatapoints 50'000

// MCTS Master parameters
#define DefaultSimulations 1600
#define DefaultEnvironments 10

// Algorithm Hyperparameters
#define ExplorationBias 0.25
#define PolicyBias 0.1
#define ValueBias 1

// ---- Performance Settings ----
#define MaxThreads 8
// These are target values, will not always be matched
// How many simulations a thread should aim to handle
#define PerThreadSimulations 256
// How many nodes a single thread should convert to gamestates
#define PerThreadGamestateConvertions 128

// Torch Settings
// This is where tensors are created and simmelar
#define TorchDefaultDevice torch::kCPU
// This is the device computations will be run on
#define TorchInferenceDevice torch::kMPS
// Floating point precision for Inference
#define TorchDefaultScalar torch::kFloat16
// Higher is better if VRAM/RAM can handle
#define MaxBatchsize 4096
// -------------------------------

// Save memory if 2d -> 1d index mapping fits in 2^8
#if Boardsize < 16
typedef uint8_t index_t;
#else
typedef uint16_t index_t;
#endif