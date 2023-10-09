#pragma once

#include "Config.h"

class Model
{
public:
    Model(torch::jit::script::Module  jit_module);
    Model(std::string model_path);

    std::tuple<torch::Tensor, float> forward(torch::Tensor input);

private:
    torch::jit::script::Module neural_network;
};