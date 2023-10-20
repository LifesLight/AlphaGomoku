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

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

#define ExplorationBias 0.225
#define PolicyBias 0.3
#define ValueBias 1
#define MaxSimulations 100'000
#define HistoryDepth 8