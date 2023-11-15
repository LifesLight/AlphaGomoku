#pragma once

// Stone skin packs
#define SKIN_DEFAULT

#ifdef SKIN_DEFAULT
#define BlackStoneUni "●"
#define WhiteStoneUni "●"
#define BlackStoneCol "\033[1;34m"
#define WhiteStoneCol "\033[1;31m"
#endif

#ifdef SKIN_NUCLEAR
#define BlackStoneUni "☢"
#define WhiteStoneUni "☢"
#define BlackStoneCol "\033[1;92m"
#define WhiteStoneCol "\033[1;93m"
#endif