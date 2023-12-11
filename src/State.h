#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include <string>
#include <vector>

#include "Config.h"
#include "Utilities.h"
#include "Types.h"


// Block is a type that is large enough to store one board row/collumn
#if BoardSize > 32
typedef i64_t block_t;
#elif BoardSize > 16
typedef i32_t block_t;
#elif BoardSize > 8
typedef i16_t block_t;
#else
typedef i8_t block_t;
#endif

/**
 * Optimized Gomoku game state interface for MCTS
 * This class is derived from my GomokuMCTS State class,
 * but has been optimized for memory usage instead of speed
 */

enum class StateResult {
    BLACKWIN,
    WHITEWIN,
    DRAW,
    NONE
};

enum class StateColor {
    BLACK,
    WHITE,
    EMPTY
};

/*
State is a highly optimized, memory efficient representation of a singular Gomoku Board.
It has fast checks for if someone has won or if the Board is terminal (No moves left or player won).

Is also the main interface for getting information about a Board.
*/
class State {
 public:
    /**
     * Default constructor. Initializes a new State object.
     */
    State();

    /**
     * Copy constructor. Initializes a new State object based on an existing one.
     * @param source The source State object to copy from.
     */
    explicit State(State* source);

    /**
     * Makes a move on the board.
     * @param index The index on the board where the move should be made.
     */
    void makeMove(index_t index);

    /**
     * Makes a move on the board.
     * @param x The x-coordinate of the move to be made.
     * @param y The y-coordinate of the move to be made.
     */
    void makeMove(u8_t x, u8_t y);

    /**
     * Returns a list of indices representing the remaining empty fields on the board.
     * @return A vector of indices for the empty fields.
     */
    std::vector<index_t> getPossible();

    /**
     * Checks if the game is in a terminal state.
     * @return True if the game is in a terminal state, false otherwise.
     */
    bool isTerminal();

    /**
     * Returns the result of the game.
     * @return A StateResult value representing the result of the game.
     */
    StateResult getResult();

    /**
     * Returns the value of a cell on the board.
     * @param index The index of the cell.
     * @return The value of the cell.
     */
    int8_t getCellValue(index_t index);

    /**
     * Returns the value of a cell on the board.
     * @param x The x-coordinate of the cell.
     * @param y The y-coordinate of the cell.
     * @return The value of the cell.
     */
    int8_t getCellValue(u8_t x, u8_t y);

    /**
     * Checks if a cell on the board is empty.
     * @param index The index of the cell.
     * @return True if the cell is empty, false otherwise.
     */
    bool isCellEmpty(index_t index);

    /**
     * Checks if a cell on the board is empty.
     * @param x The x-coordinate of the cell.
     * @param y The y-coordinate of the cell.
     * @return True if the cell is empty, false otherwise.
     */
    bool isCellEmpty(u8_t x, u8_t y);

    /**
     * Returns the color of the next move.
     * @return A StateColor value representing the color of the next move.
     */
    StateColor getNextColor();

 private:
    // The mask-bitmap of stones on the board
    block_t m_array[BoardSize];
    // The color-bitmap of the last players stones
    block_t c_array[BoardSize];
    // Last is last played move, empty is remaining empty fields
    index_t last, empty;

    /**
     * The result of the game.
     */
    StateResult result;

    /**
     * Checks if the game has been won.
     * @return True if the game has been won, false otherwise.
     */
    bool checkForWin();

    /**
     * Checks if a cell is of the active color.
     * @param x The x-coordinate of the cell.
     * @param y The y-coordinate of the cell.
     * @return True if the cell is of the active color, false otherwise.
     */
    bool cellIsActiveColor(i32_t x, i32_t y);
};
