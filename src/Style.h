#pragma once

// Stone skin packs
#define STONES_DEFAULT
#define LINES_DOUBLE

#ifdef STONES_DEFAULT
#define BlackStoneUni "●"
#define WhiteStoneUni "●"
#define BlackStoneCol "\033[1;34m"
#define WhiteStoneCol "\033[1;31m"
#endif

#ifdef STONES_NUCLEAR
#define BlackStoneUni "☢"
#define WhiteStoneUni "☢"
#define BlackStoneCol "\033[1;92m"
#define WhiteStoneCol "\033[1;93m"
#endif

#ifdef LINES_DEFAULT
#define Cornor0 "┌"
#define Cornor1 "┐"
#define Cornor2 "└"
#define Cornor3 "┘"
#define Line0 "─"
#define Line1 "│"
#define Center "┼"
#define Cross0 "┬"
#define Cross1 "├"
#define Cross2 "┤"
#define Cross3 "┴"
#endif

#ifdef LINES_BOLD
#define Cornor0 "┏"
#define Cornor1 "┓"
#define Cornor2 "┗"
#define Cornor3 "┛"
#define Line0 "━"
#define Line1 "┃"
#define Center "╋"
#define Cross0 "┳"
#define Cross1 "┣"
#define Cross2 "┫"
#define Cross3 "┻"
#endif

#ifdef LINES_DOUBLE
#define Cornor0 "╔"
#define Cornor1 "╗"
#define Cornor2 "╚"
#define Cornor3 "╝"
#define Line0 "═"
#define Line1 "║"
#define Center "╬"
#define Cross0 "╦"
#define Cross1 "╠"
#define Cross2 "╣"
#define Cross3 "╩"
#endif