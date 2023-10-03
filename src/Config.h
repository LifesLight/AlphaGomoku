#pragma once

#include "iostream"
#include "vector"
#include "list"
#include "random"
#include "algorithm"
#include "string"
#include "chrono"

#define BoardSize 15
#define ExplorationBias 1.42
#define MaxSimulations 100000000
typedef float FloatPrecision;

extern std::random_device rand_device;
extern std::mt19937 rng;
extern FloatPrecision log_table[MaxSimulations];