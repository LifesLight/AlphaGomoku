#pragma once

#include "Config.h"
#include "Node.h"

class Node;

class Gamestate
{
public:
    // From Node
    Gamestate(Node*);
    torch::Tensor getTensor();
    std::string sliceToString(uint8_t depth);

private:
    torch::Tensor tensor;
};