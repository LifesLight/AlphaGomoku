#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/

#include "Config.h"

struct Datapoint
{
    std::vector<index_t> moves;
    index_t best_move;
    // 0 is black 1 is white 2 is draw
    uint8_t winner;
};