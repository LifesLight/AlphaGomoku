#include "State.h"

State::State()
    : last(0), empty(BoardSize * BoardSize), result(2)
{
    memset(m_array, 0, sizeof(BLOCK) * BoardSize);
    memset(c_array, 0, sizeof(BLOCK) * BoardSize);
}

State::State(State* source)
    : last(source->last), empty(source->empty), result(source->result)
{
    memcpy(m_array, source->m_array, sizeof(BLOCK) * BoardSize);
    memcpy(c_array, source->c_array, sizeof(BLOCK) * BoardSize);
}

void State::makeMove(index_t index)
{
    --empty;
    last = index;
    BLOCK x, y;
    Utils::indexToCords(index, x, y);

    // Horizontal
    m_array[y] |= (BLOCK(1) << x);

    // Flip Colors
    for (index_t i = 0; i < BoardSize; i++) c_array[i] ^= m_array[i];

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
    if (m_array[y] & (BLOCK(1) << x))
    {
        if (c_array[y] & (BLOCK(1) << x))
            return empty % 2;
        return !(empty % 2);
    }

    return -1;
}

bool State::cellIsActiveColor(int x, int y)
{
    return (c_array[y] & (BLOCK(1) << x));
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
                value += Style::bsc();
                value += Style::bsu();
                value += "\033[0m";
            }
            else
            {
                value += Style::wsc();
                value += Style::wsu();
                value += "\033[0m";
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

    //Horizontal
    // This is still performant since it uses the original code for the check
    BLOCK m = c_array[y];
    m = m & (m >> BLOCK(1));
    m = (m & (m >> BLOCK(2)));
    if (m & (m >> BLOCK(1))) return true;

    // We are no longer storing representations of all directions, so this will be slow

    //Vertical
    int consecutive = 0;
    for (int i = 0; i < BoardSize; i++)
    {
        if (cellIsActiveColor(x, i))
        {
            consecutive++;
            if (consecutive == 5) return true;
        }
        else consecutive = 0;
    }

    //Diagonal
    consecutive = 0;
    int x1 = x, y1 = y;
    while (x1 > 0 && y1 > 0)
    {
        x1--;
        y1--;
    }
    while (x1 < BoardSize && y1 < BoardSize)
    {
        if (cellIsActiveColor(x1, y1))
        {
            consecutive++;
            if (consecutive == 5) return true;
        }
        else consecutive = 0;
        x1++;
        y1++;
    }

    //Anti-Diagonal
    consecutive = 0;
    x1 = x, y1 = y;
    while (x1 > 0 && y1 < BoardSize - 1)
    {
        x1--;
        y1++;
    }
    while (x1 < BoardSize && y1 >= 0)
    {
        if (cellIsActiveColor(x1, y1))
        {
            consecutive++;
            if (consecutive == 5) return true;
        }
        else consecutive = 0;
        x1++;
        y1--;
    }

    return false;
}
