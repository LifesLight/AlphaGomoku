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
};