/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "State.h"

State::State()
    : last(0), empty(BoardSize * BoardSize), result(StateResult::NONE) {
    memset(m_array, 0, sizeof(block_t) * BoardSize);
    memset(c_array, 0, sizeof(block_t) * BoardSize);
}

State::State(State* source)
    : last(source->last), empty(source->empty), result(source->result) {
    memcpy(m_array, source->m_array, sizeof(block_t) * BoardSize);
    memcpy(c_array, source->c_array, sizeof(block_t) * BoardSize);
}

void State::makeMove(u8_t x, u8_t y) {
    index_t moveIndex;
    Utils::cordsToIndex(&moveIndex, x, y);
    makeMove(moveIndex);
}

void State::makeMove(index_t index) {
    --empty;
    last = index;
    block_t x, y;
    Utils::indexToCords(index, &x, &y);

    // Horizontal
    m_array[y] |= (block_t(1) << x);

    // Flip Colors
    for (index_t i = 0; i < BoardSize; i++) c_array[i] ^= m_array[i];

    // Check for 5-Stone alignment
    bool is_won = checkForWin();
    if (is_won) {
        switch (getNextColor()) {
            case StateColor::BLACK:
                result = StateResult::BLACKWIN;
                break;
            case StateColor::WHITE:
                result = StateResult::WHITEWIN;
                break;
            default:
                break;
        }
    } else if (empty == 0) {
        result = StateResult::DRAW;
    }
}

bool State::isCellEmpty(index_t index) {
    u8_t x, y;
    Utils::indexToCords(index, &x, &y);
    return isCellEmpty(x, y);
}

bool State::isCellEmpty(u8_t x, u8_t y) {
    return !(m_array[y] & (block_t(1) << x));
}

i8_t State::getCellValue(index_t index) {
    u8_t x, y;
    Utils::indexToCords(index, &x, &y);
    return getCellValue(x, y);
}

i8_t State::getCellValue(u8_t x, u8_t y) {
    if (m_array[y] & (block_t(1) << x)) {
        if (c_array[y] & (block_t(1) << x))
            return empty % 2;
        return !(empty % 2);
    }

    return -1;
}

bool State::cellIsActiveColor(i32_t x, i32_t y) {
    return (c_array[y] & (block_t(1) << x));
}

StateColor State::getNextColor() {
    if (empty % 2)
        return StateColor::BLACK;
    return StateColor::WHITE;
}

StateResult State::getResult() {
    return result;
}

vector<index_t> State::getPossible() {
    vector<index_t> actions;
    actions.reserve(empty);
    u8_t x, y;
    for (index_t i = 0; i < BoardSize * BoardSize; i++) {
        Utils::indexToCords(i, &x, &y);
        if (isCellEmpty(x, y)) actions.push_back(i);
    }
    return actions;
}

bool State::isTerminal() {
    return (result != StateResult::NONE);
}

bool State::checkForWin() {
    u8_t x, y;
    Utils::indexToCords(last, &x, &y);

    // Horizontal
    // This is still performant since it uses the original code for the check
    block_t m = c_array[y];
    m = m & (m >> block_t(1));
    m = (m & (m >> block_t(2)));
    if (m & (m >> block_t(1))) return true;

    // Vertical
    i32_t consecutive = 0;
    for (i32_t i = 0; i < BoardSize; i++) {
        if (cellIsActiveColor(x, i)) {
            consecutive++;
            if (consecutive == 5) return true;
        } else {
            consecutive = 0;
        }
    }

    // Diagonal
    consecutive = 0;
    i32_t x1 = x, y1 = y;
    while (x1 > 0 && y1 > 0) {
        x1--;
        y1--;
    }

    while (x1 < BoardSize && y1 < BoardSize) {
        if (cellIsActiveColor(x1, y1)) {
            consecutive++;
            if (consecutive == 5) {
                return true;
            }
        } else {
            consecutive = 0;
        }
        x1++;
        y1++;
    }

    // Anti-Diagonal
    consecutive = 0;
    x1 = x, y1 = y;
    while (x1 > 0 && y1 < BoardSize - 1) {
        x1--;
        y1++;
    }

    while (x1 < BoardSize && y1 >= 0) {
        if (cellIsActiveColor(x1, y1)) {
            consecutive++;
            if (consecutive == 5) return true;
        } else {
            consecutive = 0;
        }
        x1++;
        y1--;
    }

    return false;
}

string State::str() {
    // Constants for rendering
    const string stoneBlack = " ● ";
    const string stoneWhite = " ● ";
    const string colorBlack = "\033[0;34m";
    const string colorWhite = "\033[0;31m";
    const string resetColor = "\033[0m";

    vector<vector<string>> cellValues;
    for (i32_t x = 0; x < BoardSize; x++) {
        vector<string> collumn;
        for (i32_t y = 0; y < BoardSize; y++) {
            string value;
            i8_t index_value = getCellValue(x, y);
            if (index_value == -1) {
                value += "   ";
            } else if (index_value == 0) {
                value += colorBlack;
                value += stoneBlack;
                value += resetColor;
            } else {
                value += colorWhite;
                value += stoneWhite;
                value += resetColor;
            }
            collumn.push_back(value);
        }
        cellValues.push_back(collumn);
    }

    return Utils::cellsToString(cellValues);
}
