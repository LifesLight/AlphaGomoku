#pragma once
#include "Includes.h"

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

#define ExplorationBias 1.41421356237
#define MaxSimulations 10'000'000
#define HistoryDepth 10
typedef float FloatPrecision;

extern std::random_device rand_device;
extern std::mt19937 rng;
extern FloatPrecision logTable[MaxSimulations];