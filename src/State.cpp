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

bool State::isTerminal()
{
    return (empty == 0 || result < 2);
}

std::string State::toString()
{
    std::string result;
    result += "    ";
    
    for (uint16_t i = 0; i < BoardSize; i++)
        result += " " + std::to_string(i) + std::string(3 - std::to_string(i).length(), ' ');
    
    result += "\n   ";
    
    for (uint16_t i = 0; i < BoardSize; i++)
        result += " ---";
    
    result += "\n";
    
    for (uint16_t y = 0; y < BoardSize; y++)
    {
        result += std::to_string(y) + std::string(3 - std::to_string(y).length(), ' ');
        
        for (uint16_t x = 0; x < BoardSize; x++)
        {
            result += "|";
            
            if (!(m_array[y] & (BLOCK(1) << x)))
                result += "   ";
            else if (c_array[y] & (BLOCK(1) << x))
            {
                if (!(empty % 2))
                    result += "\033[1;34m o \033[0m";
                else
                    result += "\033[1;31m o \033[0m";
            }
            else if (!(c_array[y] & (BLOCK(1) << x)))
            {
                if (empty % 2)
                    result += "\033[1;34m o \033[0m";
                else
                    result += "\033[1;31m o \033[0m";
            }
        }
        result += "|\n   ";
        
        for (uint16_t i = 0; i < BoardSize; i++)
            result += " ---";
        
        result += "\n";
    }
    
    return result;
}

bool State::checkForWin()
{
    uint8_t x = last % BoardSize;
    uint8_t y = last / BoardSize;
#if BoardSize < 16
    uint64_t m = ((uint64_t)c_array[y] << 48)
        + ((uint64_t)c_array[x + BoardSize] << 32)
        + ((uint64_t)c_array[x + BoardSize - 1 - y + BoardSize * 2] << 16)
        + ((uint64_t)c_array[BoardSize - 1 - x + BoardSize - 1 - y + BoardSize * 4]);

    m &= (m >> uint64_t(1));
    m &= (m >> uint64_t(2));
    return (m & (m >> uint64_t(1)));
#else
    //Horizontal
    BLOCK m = c_array[y];
    m = m & (m >> BLOCK(1));
    m = (m & (m >> BLOCK(2)));
    if (m & (m >> BLOCK(1))) return true;
    //Vertical
    m = c_array[x + BoardSize];
    m = m & (m >> BLOCK(1));
    m = (m & (m >> BLOCK(2)));
    if (m & (m >> BLOCK(1))) return true;
    //LDiagonal
    m = c_array[x + BoardSize - 1 - y + BoardSize * 2];
    m = m & (m >> BLOCK(1));
    m = (m & (m >> BLOCK(2)));
    if (m & (m >> BLOCK(1))) return true;
    //RDiagonal
    m = c_array[BoardSize - 1 - x + BoardSize - 1 - y + BoardSize * 4];
    m = m & (m >> BLOCK(1));
    m = (m & (m >> BLOCK(2)));
    if (m & (m >> BLOCK(1))) return true;
    return false;
#endif
}