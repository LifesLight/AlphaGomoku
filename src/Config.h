#pragma once
#include "Includes.h"

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

#define ExplorationBias 50
#define PolicyBias 0.05
#define MaxSimulations 10'000'000
#define HistoryDepth 8
typedef float FloatPrecision;

extern std::random_device rand_device;
extern std::mt19937 rng;
extern FloatPrecision logTable[MaxSimulations];