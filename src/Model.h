#pragma once

#include "Config.h"

/*
Wrapper around a pytorch multi module model.
Used for simplifying the code.
*/

class Model
{
public:
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, torch::Device device, std::string name);
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, std::string name);
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, torch::Device device);
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
    std::string getName();

    // Creates a model from just the model name, takes rest from config path
    static Model* autoloadModel(std::string name, torch::Device device);

private:
    torch::jit::script::Module resnet, polhead, valhead;
    torch::Device device;
    std::string model_name;
};
