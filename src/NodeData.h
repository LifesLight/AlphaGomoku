#pragma once

#include "Config.h"

struct NodeData
{
    uint32_t visits;
    std::deque<index_t> untried_actions;
    float evaluation;
    float summed_evaluation;
    torch::Tensor policy_evaluations;
};