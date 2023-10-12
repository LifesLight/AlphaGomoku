#pragma once

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

#define ExplorationBias 0.2
#define PolicyBias 0.5
#define MaxSimulations 100'000
#define HistoryDepth 8