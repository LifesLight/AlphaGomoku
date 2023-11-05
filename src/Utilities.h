#pragma once

#include "Config.h"
#include "Model.h"

class Utils
{
public:
    template <typename T>
    static void indexToCords(index_t index, T& x, T& y)
    {
        x = index / BoardSize;
        y = index % BoardSize;
    }


    template <typename T>
    static void cordsToIndex(index_t& index, T x, T y)
    {
        index = x * BoardSize + y;
    }

    static std::string getEnv(const std::string& variable)
    {
        const char* env_value = std::getenv(variable.c_str());
        if (env_value == nullptr)
            return "NONE";

        std::string env_value_lower = env_value;
        std::transform(env_value_lower.begin(), env_value_lower.end(), env_value_lower.begin(), ::tolower);
        return env_value_lower;
    }

    static bool checkEnv(const std::string& variable, const std::string& target)
    {
        std::string env_value_lower = getEnv(variable);

        std::string target_lower = target;
        std::transform(target_lower.begin(), target_lower.end(), target_lower.begin(), ::tolower);

        return env_value_lower == target_lower;
    }

    static std::string sliceGamestate(torch::Tensor gamestate, int depth)
    {
        int HD = HistoryDepth;
        if (depth > HD - 2)
            std::cout << "[Utilities][W]: Gamestate sliced too deep" << std::endl;

        std::string output = "";
        int halfDepth = HD / 2;
        torch::Tensor blackStones, whiteStones;

        if (HD == 2) 
        {
            blackStones = gamestate[1];
            whiteStones = gamestate[2];
        } 
        else 
        {
            int tempDepth = depth / 2;
            int whiteIndex = halfDepth * 2 - tempDepth;
            int blackIndex = halfDepth - tempDepth;
            if (depth % 2 == 1) {
                if (gamestate[0][0][0].item<bool>() == 1)
                    blackIndex -= 1;
                else
                    whiteIndex -= 1;
            }
            blackStones = gamestate[blackIndex];
            whiteStones = gamestate[whiteIndex];
        }

        output += "\n   ";
        for (int i = 0; i < BoardSize; i++)
            output += " ---";
        output += "\n";

        for (int y = 14; y >= 0; y--) 
        {
            output += std::to_string(y) + std::string(3 - std::to_string(y).length(), ' ');
            for (int x = 0; x < 15; x++) 
            {
                output += "|";
                if (blackStones[x][y].item<bool>() == 0 && whiteStones[x][y].item<bool>() == 0)
                    output += "   ";
                else if (blackStones[x][y].item<bool>() == 1)
                    output += "\033[1;34m B \033[0m";
                else if (whiteStones[x][y].item<bool>() == 1)
                    output += "\033[1;31m W \033[0m";
            }
            output += "|\n   ";
            for (int i = 0; i < BoardSize; i++)
                output += " ---";
            output += "\n";
        }

        output += "    ";
        for (int i = 0; i < BoardSize; i++)
            output += " " + std::to_string(i) + std::string(3 - std::to_string(i).length(), ' ');
        output += "\n";

        return output;
    }
};