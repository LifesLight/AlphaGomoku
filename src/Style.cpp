#include "Style.h"

Stone Style::selected_stones = Stone::DEFAULT;
Board Style::selected_board = Board::DEFAULT;

std::map<std::string, Stone> Style::stone_to_enum  = 
{
    {"default", Stone::DEFAULT},
    {"nuclear", Stone::NUCLEAR},
    {"animals", Stone::ANIMALS},
};

std::map<std::string, Board> Style::board_to_enum  = 
{
    {"default", Board::DEFAULT},
    {"bold", Board::BOLD},
    {"double", Board::DOUBLE},
};

void Style::setStone(std::string stone)
{
    if (stone_to_enum.find(stone) != stone_to_enum.end())
    {
        selected_stones = stone_to_enum[stone];
    }
    else
    {
        Log::log(LogLevel::WARNING, "Tried to assign unknown Stone skin, falling back to default", "STYLE");
    }
}

void Style::setBoard(std::string board)
{
    if (board_to_enum.find(board) != board_to_enum.end())
    {
        selected_board = board_to_enum[board];
    }
    else
    {
        Log::log(LogLevel::WARNING, "Tried to assign unknown Board skin, falling back to default", "STYLE");
    }
}

// Black Stone
std::string Style::bsu()
{
    switch(selected_stones)
    {
        case Stone::DEFAULT:
            return " ● ";
        case Stone::NUCLEAR:
            return " ☢ ";
        case Stone::ANIMALS:
            return " 😾";
    }

    return "BSU";
}

// White Stone
std::string Style::wsu()
{
    switch(selected_stones)
    {
        case Stone::DEFAULT:
            return " ● ";
        case Stone::NUCLEAR:
            return " ☢ ";
        case Stone::ANIMALS:
            return " 🙈";
    }

    return "WSU";
}

// Black Stone Color
std::string Style::bsc()
{
    switch(selected_stones)
    {
        case Stone::DEFAULT:
            return "\033[1;34m";
        case Stone::NUCLEAR:
            return "\033[1;92m";
        case Stone::ANIMALS:
            return "\033[1;34m";
    }

    return "";
}

// White Stone Color
std::string Style::wsc()
{
    switch(selected_stones)
    {
        case Stone::DEFAULT:
            return "\033[1;31m";
        case Stone::NUCLEAR:
            return "\033[1;93m";
        case Stone::ANIMALS:
            return "\033[1;31m";
    }

    return "";
}

// Grid drawing lines
std::string Style::cornor0()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┌";
        case Board::BOLD:
            return "┏";
        case Board::DOUBLE:
            return "╔";
    }

    return "+";
}
std::string Style::cornor1()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┐";
        case Board::BOLD:
            return "┓";
        case Board::DOUBLE:
            return "╗";
    }

    return "+";
}
std::string Style::cornor2()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "└";
        case Board::BOLD:
            return "┗";
        case Board::DOUBLE:
            return "╚";
    }

    return "+";
}
std::string Style::cornor3()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┘";
        case Board::BOLD:
            return "┛";
        case Board::DOUBLE:
            return "╝";
    }

    return "+";
}
std::string Style::line0()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "─";
        case Board::BOLD:
            return "━";
        case Board::DOUBLE:
            return "═";
    }

    return "-";
}
std::string Style::line1()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "│";
        case Board::BOLD:
            return "┃";
        case Board::DOUBLE:
            return "║";
    }

    return "|";
}
std::string Style::center()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┼";
        case Board::BOLD:
            return "╋";
        case Board::DOUBLE:
            return "╬";
    }

    return "+";
}
std::string Style::cross0()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┬";
        case Board::BOLD:
            return "┳";
        case Board::DOUBLE:
            return "╦";
    }

    return "+";
}
std::string Style::cross1()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "├";
        case Board::BOLD:
            return "┣";
        case Board::DOUBLE:
            return "╠";
    }

    return "+";
}
std::string Style::cross2()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┤";
        case Board::BOLD:
            return "┫";
        case Board::DOUBLE:
            return "╣";
    }

    return "+";
}
std::string Style::cross3()
{
    switch(selected_board)
    {
        case Board::DEFAULT:
            return "┴";
        case Board::BOLD:
            return "┻";
        case Board::DOUBLE:
            return "╩";
    }

    return "+";
}