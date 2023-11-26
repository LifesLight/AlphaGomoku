#pragma once

#include "Config.h"
#include "Log.h"

enum class Stone {DEFAULT, NUCLEAR, ANIMALS};
enum class Board {DEFAULT, BOLD, DOUBLE};

class Style
{
private:
    static enum Stone selected_stones;
    static enum Board selected_board;

public:
    static void setStone(std::string stone);
    static void setBoard(std::string board);

    // Get black/white stone unicode/color
    static std::string bsu();
    static std::string wsu();
    static std::string bsc();
    static std::string wsc();

    // Get all grid line types
    static std::string cornor0();
    static std::string cornor1();
    static std::string cornor2();
    static std::string cornor3();
    static std::string line0();
    static std::string line1();
    static std::string center();
    static std::string cross0();
    static std::string cross1();
    static std::string cross2();
    static std::string cross3();

private:
    static std::map<std::string, Stone> stone_to_enum;
    static std::map<std::string, Board> board_to_enum;

    // Prevent instantiation
    Style() = delete;
};