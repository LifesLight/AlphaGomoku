#include "State.h"

State::State()
    : last(0), empty(BoardSize * BoardSize), result(2)
{
    memset(m_array, 0, sizeof(BLOCK) * BoardSize * 6);
    memset(c_array, 0, sizeof(BLOCK) * BoardSize * 6);
}

State::State(State* source)
    : last(source->last), empty(source->empty), result(source->result)
{
    memcpy(m_array, source->m_array, sizeof(BLOCK) * BoardSize * 6);
    memcpy(c_array, source->c_array, sizeof(BLOCK) * BoardSize * 6);
}

void State::makeMove(uint16_t index)
{
    --empty;
    last = index;
    BLOCK x = index % BoardSize;
    BLOCK y = index / BoardSize;
    // Horizontal
    m_array[y] |= (BLOCK(1) << x);
    // Vertical
    m_array[x + BoardSize] |= (BLOCK(1) << y);
    // LDiagonal
    m_array[x + BoardSize - 1 - y + BoardSize * 2] |= (BLOCK(1) << x);
    // RDiagonal
    m_array[BoardSize - 1 - x + BoardSize - 1 - y + BoardSize * 4] |= (BLOCK(1) << x);
    // Flip Colors
    for (uint16_t i = 0; i < BoardSize * 6; i++) c_array[i] ^= m_array[i];

    // Check for 5-Stone alignment
    result = checkForWin() ? empty % 2 : 2;
}

std::vector<uint16_t> State::getPossible()
{
    std::vector<uint16_t> actions;
    actions.reserve(empty);
    for (uint16_t i = 0; i < BoardSize * BoardSize; i++)
        if (!(m_array[i / BoardSize] & (BLOCK(1) << i % BoardSize))) actions.push_back(i);
    return actions;
}