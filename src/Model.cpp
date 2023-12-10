/**
 * Copyright (c) Alexander Kurtz 2023
*/


#include "Model.h"

torch::jit::script::Module Model::load_module(std::string path)
{
    // Always load on CPU
    torch::jit::script::Module model = torch::jit::load(path, Config::torchHostDevice());
    model.to(Config::torchInferenceDevice());
    model.to(Config::torchScalar());
    model.eval();
    return model;
}

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path)
    : Model(resnet_path, polhead_path, valhead_path, Config::defaultSimulations())
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, int simulations)
    : Model(resnet_path, polhead_path, valhead_path, simulations, "unnamed")
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, std::string name)
    : Model(resnet_path, polhead_path, valhead_path, Config::defaultSimulations(), name)
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, int simulations, std::string name)
    : model_name(name), simulations(simulations), device(Config::torchInferenceDevice()), dtype(Config::torchScalar())
{
    // Load resnet
    try
    {
        resnet = load_module(resnet_path);
    }
    catch (const c10::Error& e)
    {
        Log::log(LogLevel::FATAL, "Could not load resnet from: " + resnet_path, "MODEL");
    }

    // Load polhead
    try
    {
        polhead = load_module(polhead_path);
    }
    catch (const c10::Error& e)
    {
        Log::log(LogLevel::FATAL, "Could not load policyhead from: " + polhead_path, "MODEL");
    }

    // Load valhead
    try
    {
        valhead = load_module(valhead_path);
    }
    catch (const c10::Error& e)
    {
        Log::log(LogLevel::FATAL, "Could not load valuehead from: " + valhead_path, "MODEL");
    }
}


std::tuple<torch::Tensor, torch::Tensor> Model::forward(torch::Tensor input)
{
    // Disable gradients for this scope
    torch::NoGradGuard no_grad_guard;

    // Inference
    auto resnet_result = resnet.forward({input});
    auto policy_result = polhead.forward({resnet_result});
    auto value_result = valhead.forward({resnet_result});

    // Extract policy and value outputs
    torch::Tensor policy_output = policy_result.toTensor();
    torch::Tensor value_output = value_result.toTensor();
    policy_output = torch::softmax(policy_output, -1);
    policy_output = policy_output.to(Config::torchHostDevice());
    value_output = value_output.to(Config::torchHostDevice());

    // Detach for grad safety
    policy_output = policy_output.detach();
    value_output = value_output.detach();

    return std::tuple<torch::Tensor, torch::Tensor>(policy_output, value_output);
}

void Model::setDevice(torch::Device device)
{
    this->device = device;
    resnet.to(device);
    valhead.to(device);
    polhead.to(device);
}

torch::Device Model::getDevice()
{
    return device;
}

void Model::setPrec(torch::ScalarType type)
{
    dtype = type;
    resnet.to(type);
    valhead.to(type);
    polhead.to(type);
}

torch::ScalarType Model::getPrec()
{
    return dtype;
}

void Model::setName(std::string name)
{
    model_name = name;
}

std::string Model::getName()
{
    std::string name = model_name;
    name += " (";
    name += std::to_string(getSimulations());
    name += "|";
    name += torch::toString(dtype);
    name += "|";
    name += c10::DeviceTypeName(device.type());
    name += ")";
    return name;
}

void Model::setSimulations(int sims)
{
    simulations = sims;
}

int Model::getSimulations()
{
    return simulations;
}

Model* Model::autoloadModel(std::string name)
{
    return autoloadModel(name, Config::defaultSimulations());
}

Model* Model::autoloadModel(std::string name, int simulations)
{
    Log::log(LogLevel::INFO, "Autoloading: " + name, "MODEL");

    std::string general_path = Config::modelPath();
    std::string resnet_path = general_path + "ResNet/" + name;
    std::string policy_path = general_path + "PolHead/" + name;
    std::string value_path = general_path + "ValHead/" + name;

    Model* loaded_model = nullptr;
    try
    {
        loaded_model = new Model(resnet_path, policy_path, value_path, simulations, name);
    }
    catch(const std::exception& e)
    {
        Log::log(LogLevel::ERROR, "Failed to autoload Model", "MODEL");
    }
    return loaded_model;
}