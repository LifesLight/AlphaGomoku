#pragma once

#include "Utilities.h"
#include "Node.h"
#include "Model.h"
#include "State.h"

class Environment
{
public:
    Environment(Model* NNB, Model* NNW);
    // Returns success
    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(uint16_t index);

private:
    State* current_state;
    Node* root_node[2];
    Node* current_node[2];
    Model* neural_network[2];
    bool is_ai[2];
    uint16_t total_moves;
    float log_table[MaxSimulations];
};