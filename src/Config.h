#pragma once

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

#define ExplorationBias 1.25
#define PolicyBias 0.05
#define MaxSimulations 10'000'000
#define HistoryDepth 8