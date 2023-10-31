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

// CPU Threads
#define ThreadCount 1

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

// HD
#define HistoryDepth 8

#define TorchDevice torch::kCUDA

// Algorithm Hyperparameters
#define ExplorationBias 0.2
#define PolicyBias 0.3
#define ValueBias 1
