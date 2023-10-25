#include "Model.h"

torch::jit::script::Module load_model(std::string path, torch::Device device)
{
    // Always load on CPU
    torch::jit::script::Module model = torch::jit::load(path, torch::kCPU);
    model.to(device);
    model.eval();  
    return model;  
}

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path)
    : Model(resnet_path, polhead_path, valhead_path, torch::kCPU)
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, torch::Device device)
    : Model(resnet_path, polhead_path, valhead_path, device, "unnamed")
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, torch::Device device, std::string name)
    : device(device), model_name(name)
{
    // Load resnet
    try
    {
        resnet = load_model(resnet_path, device);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load resnet from:" << std::endl << resnet_path << std::endl;
    }

    // Load polhead
    try
    {
        polhead = load_model(polhead_path, device);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load policyhead from:" << std::endl << polhead_path << std::endl;
    }

    // Load valhead
    try
    {
        valhead = load_model(valhead_path, device);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load valuehead from:" << std::endl << valhead_path << std::endl;
    }
}


std::tuple<torch::Tensor, torch::Tensor> Model::forward(torch::Tensor input)
{
    input = input.to(device);
    auto resnet_result = resnet.forward({input});
    auto policy_result = polhead.forward({resnet_result});
    auto value_result = valhead.forward({resnet_result});

    // Extract policy and value outputs
    torch::Tensor policy_output = policy_result.toTensor();
    torch::Tensor value_output = value_result.toTensor();
    policy_output = torch::softmax(policy_output, -1);

    return std::tuple<torch::Tensor, torch::Tensor>(policy_output, value_output);
}

std::string Model::getName()
{
    return model_name;
}