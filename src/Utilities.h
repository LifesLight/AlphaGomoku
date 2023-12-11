#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include "Config.h"
#include "Types.h"
#include "Log.h"

class Utils {
 public:
    template <typename T>
    static void indexToCords(const index_t &index, T *x, T *y) {
        *x = index / BoardSize;
        *y = index % BoardSize;
    }


    template <typename T>
    static void cordsToIndex(index_t *index, const T &x, const T &y) {
        *index = x * BoardSize + y;
    }

    template <typename T>
    static void eraseFromVector(
        const std::vector<T>& vector,
        const T& element) {
            vector.erase(
                std::remove(
                    vector.begin(),
                    vector.end(),
                    element),
                vector.end());
    }

    static void keyboardCordsInput(index_t *x, index_t *y) {
        while (true) {
            std::string input;

            std::cout << "[Move]: ";
            std::getline(std::cin, input);

            size_t pos = input.find_first_of(",; ");

            if (pos == std::string::npos) {
                Log::log(LogLevel::ERROR,
                    "Invalid input format for cord input! Usage: x,y",
                    "UTILITIES");
                continue;
            }

            std::string x_s = input.substr(0, pos);
            std::string y_s = input.substr(pos + 1);

            try {
                *x = std::stoi(x_s);
                *y = std::stoi(y_s);

                break;
            }
            catch(const std::exception& e) {
                Log::log(LogLevel::ERROR,
                    "Invalid format for cord input! Usage: x,y",
                    "UTILITIES");
            }
        }
    }

    static std::map<std::string, std::string>
        parseArgv(i32_t argc, const char* argv[]) {
        std::map<std::string, std::string> args;

        for (i32_t i = 0; i < argc; i++) {
            std::string arg = argv[i];
            if (arg.find("--") == std::string::npos)
                continue;

            std::string key = arg.substr(2);
            std::string value = (i + 1 < argc) ? argv[i + 1] : "default_value";

            // Make keys and values lowercase
            std::transform(key.begin(),
                key.end(),
                key.begin(),
                ::tolower);
            std::transform(value.begin(),
                value.end(),
                value.begin(),
                ::tolower);

            args[key] = value;
        }

        return args;
    }
};
