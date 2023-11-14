#pragma once

#include "Config.h"

// Stores data about a node, which won't be needed in cold tree
struct NodeData
{
    uint32_t visits;
    std::deque<index_t> untried_actions;
    float evaluation;
    float summed_evaluation;
    torch::Tensor policy_evaluations;
};