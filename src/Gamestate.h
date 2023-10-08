#pragma once

#include "Includes.h"
#include "Node.h"

class Gamestate
{
public:
    // From Node
    Gamestate(Node*);

private:
    torch::Tensor tensor;
};