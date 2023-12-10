#pragma once

/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Config.h"
#include "Utilities.h"
#include "Log.h"

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
    
    // Name config
    void setName(std::string);
    std::string getName();
    
    // Device config
    void setDevice(torch::Device device);
    torch::Device getDevice();

    // Precision config
    void setPrec(torch::ScalarType type);
    torch::ScalarType getPrec();

    // Simulation config
    void setSimulations(int simulations);
    int getSimulations();

    // Creates a model from just the model name, takes rest from config path with 400 sims
    static Model* autoloadModel(std::string name);

    // Creates a model from just the model name, takes rest from config path
    static Model* autoloadModel(std::string name, int simulations);

private:
    torch::jit::script::Module load_module(std::string path);

    std::string model_name;
    int simulations;

    torch::Device device;
    torch::ScalarType dtype;

    torch::jit::script::Module resnet, polhead, valhead;
};
