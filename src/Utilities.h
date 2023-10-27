#pragma once

#include "Config.h"
#include "Model.h"

class Utils
{
public:
    template <typename T>
    static void indexToCords(uint16_t index, T& x, T& y)
    {
        x = index / BoardSize;
        y = index % BoardSize;
    }


    template <typename T>
    static void cordsToIndex(uint16_t& index, T x, T y)
    {
        index = x * BoardSize + y;
    }
};