#pragma once

#include "Config.h"
#include "Utilities.h"
#include "Style.h"

// Optimized Gomoku game state interface for MCTS
// This class is derived from my GomokuMCTS State class, but has been optimized for memory usage instead of speed

#if BoardSize > 32
typedef int64_t BLOCK;
#elif BoardSize > 16
typedef int32_t BLOCK;
#elif BoardSize > 8
typedef int16_t BLOCK;
#else
typedef int8_t BLOCK;
#endif

/*
State is a highly optimized representation of a singular Gomoku Board.
It has fast checks for if someone has one or if the Board is terminal (No moves left or player won).

Is also the main interface for getting information about a Board.
*/

enum class StateResult
{
    BLACKWIN,
    WHITEWIN,
    DRAW,
    NONE
};

class State
{
public:
    // Mask
    BLOCK m_array[BoardSize];
    // Color
    BLOCK c_array[BoardSize];
    // Last is last played move, empty is remaining empty fields
    index_t last, empty;

    State();
    State(State*);

    // Make move
    void makeMove(index_t);
    // Get list of remaining empty fields as indecies
    std::vector<index_t> getPossible();
    // Is terminal game state
    bool isTerminal();
    // Black is 0 White is 1 Draw is 2
    StateResult getResult();
    // String representation of state
    std::string toString();
    // Value of field
    int8_t getCellValue(index_t index);
    int8_t getCellValue(uint8_t x, uint8_t y);

    bool isCellEmpty(index_t index);
    bool isCellEmpty(uint8_t x, uint8_t y);

    bool getNextColor();

private:
    StateResult result;

    bool checkForWin();
    bool cellIsActiveColor(int x, int y);
};