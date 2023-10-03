#include "Config.h"

std::random_device rand_device;
std::mt19937 rng(rand_device());
FloatPrecision log_table[MaxSimulations];