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

/**
 * Utility class for various functions
*/
class Utils {
 public:
    /**
     * Converts a 2D index to 2D coordinates.
     * @param index The index to convert.
     * @param x The x coordinate to write to.
     * @param y The y coordinate to write to.
    */
    template <typename T>
    static void indexToCords(const index_t &index, T *x, T *y) {
        *x = index / BoardSize;
        *y = index % BoardSize;
    }

    /**
     * Converts 2D coordinates to a 2D index.
     * @param index The index to write to.
     * @param x The x coordinate to convert.
     * @param y The y coordinate to convert.
    */
    template <typename T>
    static void cordsToIndex(index_t *index, const T &x, const T &y) {
        *index = x * BoardSize + y;
    }

    /**
     * Erases an element from a vector.
     * @param vector The vector to erase from.
     * @param element The element to erase.
     * @tparam T The type of the vector.
    */
    template <typename T>
    static void eraseFromVector(
        const std::vector<T>& vector,
        const T& element) {
            vector.erase(
                std::remove(
                    vector.begin,
                    vector.end,
                    element),
                vector.end);
    }

    /**
     * Gets keyboard input for x y coordinates.
     * @param x The x coordinate to write to.
     * @param y The y coordinate to write to.
    */
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

    /**
     * Converts a vector of cell values to a string representation of the board.
     * @param cellValues A vector of cell values.
     * @return A string representation of the board.
    */
    static std::string cellsToString(
        const std::vector<std::vector<std::string>>& cellValues) {
        // Constants for style render style
        const std::string cornor0 = "╔";
        const std::string cornor1 = "╗";
        const std::string cornor2 = "╚";
        const std::string cornor3 = "╝";
        const std::string line0 = "═";
        const std::string line1 = "║";
        const std::string center = "╬";
        const std::string cross0 = "╦";
        const std::string cross1 = "╠";
        const std::string cross2 = "╣";
        const std::string cross3 = "╩";

        // Top line
        std::stringstream output;
        std::string three_lines = "";
        for (int i = 0; i < 3; i++)
            three_lines += line0;

        output << "   " << cornor0;
        for (int i = 0; i < BoardSize - 1; i++) {
            output << three_lines << cross0;
        }
        output << three_lines << cornor1;
        output << std::endl;

        // Inner lines
        for (int16_t y = BoardSize - 1; y >= 0; y--) {
            // Data line
            output << std::to_string(y);
            output << std::string(3 - std::to_string(y).length(), ' ');

            for (int16_t x = 0; x < BoardSize; x++) {
                output << line1;
                output << cellValues[x][y];
            }
            output << line1;
            output << std::endl;

            // Row line
            if (y == 0)
                continue;

            output << "   " << cross1;
            for (int i = 0; i < BoardSize - 1; i++) {
                output << three_lines << center;
            }
            output << three_lines << cross2;
            output << std::endl;
        }

        // Bottom line
        output << "   " << cornor2;
        for (int i = 0; i < BoardSize - 1; i++) {
            output << three_lines << cross3;
        }
        output << three_lines << cornor3;
        output << std::endl;

        output << "    ";
        for (int i = 0; i < BoardSize; i++) {
            output << " ";
            output << std::to_string(i);
            output << std::string(3 - std::to_string(i).length(), ' ');
        }

        return output.str();
    }
};
