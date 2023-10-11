#pragma once

#include "Includes.h"
#include "Config.h"

class Model
{
public:
    Model(torch::jit::script::Module  jit_module, torch::Device device, std::string name);
    Model(torch::jit::script::Module  jit_module, std::string name);
    Model(torch::jit::script::Module  jit_module, torch::Device device);
    Model(torch::jit::script::Module  jit_module);


    Model(std::string model_path, torch::Device device, std::string name);
    Model(std::string model_path, std::string name);
    Model(std::string model_path, torch::Device device);
    Model(std::string model_path);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
    std::string getName();

private:
    torch::jit::script::Module neural_network;
    torch::Device device;
    std::string model_name;
};
