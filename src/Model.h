#pragma once

#include "Config.h"
#include "Utilities.h"

/*
Wrapper around a pytorch multi module model.
Used for simplifying the code.
*/

class Model
{
public:
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, int simulations, std::string name);
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, int simulations);
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, std::string name);
    Model(std::string resnet_path, std::string polhead_path, std::string valhead_path);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input);
    std::string getName();
    void setName(std::string);
    int getSimulations();

    // Creates a model from just the model name, takes rest from config path with 400 sims
    static Model* autoloadModel(std::string name);

    // Creates a model from just the model name, takes rest from config path
    static Model* autoloadModel(std::string name, int simulations);

private:
    int simulations;
    torch::jit::script::Module resnet, polhead, valhead;
    std::string model_name;
};
