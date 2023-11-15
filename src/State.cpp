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

void State::makeMove(index_t index)
{
    --empty;
    last = index;
    BLOCK x, y;
    Utils::indexToCords(index, x, y);

    // Horizontal
    m_array[y] |= (BLOCK(1) << x);
    // Vertical
    m_array[x + BoardSize] |= (BLOCK(1) << y);
    // LDiagonal
    m_array[x + BoardSize - 1 - y + BoardSize * 2] |= (BLOCK(1) << x);
    // RDiagonal
    m_array[BoardSize - 1 - x + BoardSize - 1 - y + BoardSize * 4] |= (BLOCK(1) << x);
    // Flip Colors
    for (index_t i = 0; i < BoardSize * 6; i++) c_array[i] ^= m_array[i];

    // Check for 5-Stone alignment
    result = checkForWin() ? empty % 2 : 2;
}

bool State::isCellEmpty(index_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return isCellEmpty(x, y);
}

bool State::isCellEmpty(uint8_t x, uint8_t y)
{
    return !(m_array[y] & (BLOCK(1) << x));
}

int8_t State::getCellValue(index_t index)
{
    uint8_t x, y;
    Utils::indexToCords(index, x, y);
    return getCellValue(x, y);
}

int8_t State::getCellValue(uint8_t x, uint8_t y)
{
    if (empty % 2)
    {
        if (m_array[y] & (BLOCK(1) << x))
        {
            if (c_array[y] & (BLOCK(1) << x))
                return 1;
            return 0;
        }
    }

    if (m_array[y] & (BLOCK(1) << x))
    {
        if (c_array[y] & (BLOCK(1) << x))
            return 0;
        return 1;
    }

    return -1;
}

bool State::getNextColor()
{
    return !(empty % 2);
}

uint8_t State::getResult()
{
    return result;
}

std::vector<index_t> State::getPossible()
{
    std::vector<index_t> actions;
    actions.reserve(empty);
    uint8_t x, y;
    for (index_t i = 0; i < BoardSize * BoardSize; i++)
    {
        Utils::indexToCords(i, x, y);
        if (isCellEmpty(x, y)) actions.push_back(i);
    }
    return actions;
}

bool State::isTerminal()
{
    return (empty == 0 || result < 2);
}

std::string State::toString()
{
    std::stringstream result;
    std::vector<std::vector<std::string>> values;

    for (int x = 0; x < BoardSize; x++)
    {
        std::vector<std::string> collumn;
        for (int y = 0; y < BoardSize; y++)
        {
            std::string value;
            int8_t index_value = getCellValue(x ,y);
            if (index_value == -1)
                value += "   ";
            else if (index_value == 0)
            {
                value += BlackStoneCol;
                value += " ";
                value += BlackStoneUni;
                value += " \033[0m";
            }
            else
            {
                value += WhiteStoneCol;
                value += " ";
                value += WhiteStoneUni;
                value += " \033[0m";
            }
            collumn.push_back(value);
        }
        values.push_back(collumn);
    }

    result << std::endl;
    result << Utils::renderGamegrid(values);
    result << std::endl;

    return result.str();
}

bool State::checkForWin()
{
    uint8_t x, y;
    Utils::indexToCords(last, x, y);

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
