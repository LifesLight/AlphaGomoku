#pragma once

#include "Config.h"
#include "Model.h"
#include "Style.h"

#define ForcePrintln(string) std::cout << string << std::endl << std::flush
#define ForcePrint(string) std::cout << string << std::flush

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

    template <typename T>
    static void eraseFromVector(std::vector<T>& vector, T& element)
    {
        vector.erase(
            std::remove(vector.begin(), vector.end(), element), vector.end()
        );
    }

    static std::string renderGamegrid(std::vector<std::vector<std::string>>& field_values)
    {
        // Top line
        std::stringstream output;
        std::string three_lines = "";
        for (int i = 0; i < 3; i++)
            three_lines += Line0;

        output << "   " << Cornor0;
        for (int i = 0; i < BoardSize - 1; i++)
        {
            output << three_lines << Cross0;
        }
        output << three_lines << Cornor1;
        output << std::endl;

        // Inner lines
        for (int16_t y = BoardSize - 1; y >= 0; y--)
        {
            // Data line
            output << std::to_string(y) + std::string(3 - std::to_string(y).length(), ' ');

            for (int16_t x = 0; x < BoardSize; x++)
            {
                output << Line1;
                output << field_values[x][y];
            }
            output << Line1;
            output << std::endl;

            // Row line
            if (y == 0)
                continue;

            output << "   " << Cross1;
            for (int i = 0; i < BoardSize - 1; i++)
            {
                output << three_lines << Center;
            }
            output << three_lines << Cross2;
            output << std::endl;
        }

        // Bottom line
        output << "   " << Cornor2;
        for (int i = 0; i < BoardSize - 1; i++)
        {
            output << three_lines << Cross3;
        }
        output << three_lines << Cornor3;
        output << std::endl;

        output << "    ";
        for (int i = 0; i < BoardSize; i++)
            output << " " + std::to_string(i) + std::string(3 - std::to_string(i).length(), ' ');

        return output.str();
    }

    static std::string sliceGamestate(torch::Tensor gamestate, int depth)
    {
        int HD = HistoryDepth;
        if (depth > HD - 2)
            std::cout << "[Utilities][W]: Gamestate sliced too deep" << std::endl;

        std::stringstream output;
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

        output << std::endl;

        std::vector<std::vector<std::string>> field_values;
        for (int x = 0; x < BoardSize; x++) 
        {
            std::vector<std::string> line;
            for (int y = 0; y < BoardSize; y++) 
            {
                std::string value;
                if (blackStones[x][y].item<bool>() == 0 && whiteStones[x][y].item<bool>() == 0)
                    line.push_back("   ");
                else if (blackStones[x][y].item<bool>() == 1)
                {
                    value += BlackStoneCol;
                    value += " ";
                    value += BlackStoneUni;
                    value += " \033[0m";
                    line.push_back(value);
                }
                else if (whiteStones[x][y].item<bool>() == 1)
                {
                    value += WhiteStoneCol;
                    value += " ";
                    value += WhiteStoneUni;
                    value += " \033[0m";
                    line.push_back(value);
                }
            }
            field_values.push_back(line);
        }

        output << Utils::renderGamegrid(field_values);
        return output.str();
    }

    static void keyboardCordsInput(index_t& x, index_t& y)
    {
        while (true)
        {
            std::string input;

            std::cout << "[Move]: ";
            std::getline(std::cin, input);

            size_t pos = input.find_first_of(",; ");

            if (pos == std::string::npos)
            {
                ForcePrintln("[Utils][E]: Invalid input format for cord input! Usage: x,y");
                continue;
            }

            std::string x_s = input.substr(0, pos);
            std::string y_s = input.substr(pos + 1);

            try
            {
                x = std::stoi(x_s);
                y = std::stoi(y_s);

                break;
            }
            catch(const std::exception& e)
            {
                ForcePrintln("[Utils][E]: Invalid input format for cord input! Usage: x,y");
            }
        }
    }
};