#pragma once

#include "Utilities.h"
#include "Node.h"
#include "Model.h"
#include "State.h"
#include "Gamestate.h"

class Environment
{
public:
    static void initialize();

    Environment(Model* NNB, Model* NNW);
    ~Environment();
    // Returns success
    bool makeMove(uint8_t x, uint8_t y);
    bool makeMove(uint16_t index);

    // Let network compute next move
    uint16_t calculateNextMove(uint32_t simulations);

    // Display state
    std::string toString();
    std::string toString(uint8_t depth);

    bool isFinished();
    bool getNextColor();

    // Default current
    Node* getNode();
    Node* getNode(bool color);

    static std::string nodeAnalytics(Node* node, const std::initializer_list<std::string> args);  

private:
    State* current_state = nullptr;
    Node* root_node[2] = {nullptr, nullptr};
    Node* current_node[2] = {nullptr, nullptr};
    Model* neural_network[2] = {nullptr, nullptr};
    bool is_ai[2] = {false, false};

    // Static
    static float* log_table;
    static bool is_initialized;

};