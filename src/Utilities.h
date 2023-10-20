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

    static Model* autoloadModel(std::string name, torch::Device device)
    {
        std::string general_path = ModelPath;
        std::string resnet_path = general_path + "ResNet/" + name;
        std::string policy_path = general_path + "PolHead/" + name;
        std::string value_path = general_path + "ValHead/" + name;

        Model* loaded_model;
        try
        {
            loaded_model = new Model(resnet_path, policy_path, value_path, device, name);
        }
        catch(const std::exception& e)
        {
            std::cerr << "[Utils]: Failed to autoload Model" << '\n' << e.what() << '\n';
        }
        return loaded_model;
    }
};