#pragma once

#include "Config.h"

class Model
{
public:
    Model(torch::jit::script::Module  jit_module, torch::Device device);
    Model(std::string model_path, torch::Device device);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input);

private:
    torch::jit::script::Module neural_network;
    torch::Device device;
};