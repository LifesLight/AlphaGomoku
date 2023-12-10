#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/

// Even numbers in BoardSize will break State due to inverted colors!
#define BoardSize 15

/* -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#- */

// Standard Paths
#define ModelPath "../Models/scripted/"
#define DatapointPath "../Datasets/Selfplay/data.txt"
#define TreesPath "../Trees/"

// At least 2 and even number
#define HistoryDepth 8

// Algorithm Hyperparameters
#define ExplorationBias 1
#define PolicyBias 1
#define ValueBias 2
