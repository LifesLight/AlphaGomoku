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
#include <sstream>
#include <initializer_list>
#include <random>
#include <thread>

// External data paths
#define ModelPath "../Models/scripted/"
#define DatapointPath "../Datasets/Selfplay/data.txt"

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

// HD
#define HistoryDepth 8

// How many datapoints should be stored in a selfplay dataset
#define MaxDatapoints 50'000

// Default Device for torch computations
#define TorchDevice torch::kCPU

// Algorithm Hyperparameters
#define ExplorationBias 0.25
#define PolicyBias 0.3
#define ValueBias 1

// Save memory if 2d -> 1d index mapping fits in 2^8
#if Boardsize < 16
typedef uint8_t index_t;
#else
typedef uint16_t index_t;
#endif