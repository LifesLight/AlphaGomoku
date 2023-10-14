#pragma once

#include "Utilities.h"
#include "Node.h"
#include "Model.h"
#include "State.h"
#include "Gamestate.h"
#include "Tree.h"

class Environment
{
public:
    static void initialize();
    static void deinitialize();

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
    Node* getPlayingNode();
    Node* getPlayedNode();

    void freeMemory();

private:
    Tree* trees[2] = {nullptr, nullptr};
    bool next_color;

    Node* getAnyCurrentNode();

    // Static
    static float* log_table;
    static bool is_initialized;
};