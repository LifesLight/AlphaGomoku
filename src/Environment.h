#pragma once

#include "Utilities.h"
#include "Node.h"
#include "Model.h"
#include "State.h"

class Environment
{
public:
    Environment(Model* NNB, Model* NNW);
    ~Environment();
    // Returns success
    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(uint16_t index);

    // Let network compute next move
    uint16_t calculateNextMove(uint32_t simulations);

    // Display state
    std::string toString();

    bool isFinished();

    Node* getNode(bool color);

private:
    State* current_state;
    Node* root_node[2];
    Node* current_node[2];
    Model* neural_network[2];
    bool is_ai[2] = {false, false};
    float* log_table = new float[MaxSimulations];
};