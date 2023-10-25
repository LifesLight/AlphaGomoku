#include "Model.h"

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
        resnet = torch::jit::load(resnet_path);
        resnet.to(device);
        resnet.eval();
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load resnet from:" << std::endl << resnet_path << std::endl;
    }

    // Load polhead
    try
    {
        polhead = torch::jit::load(polhead_path);
        polhead.to(device);
        polhead.eval();
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load policyhead from:" << std::endl << polhead_path << std::endl;
    }

    // Load valhead
    try
    {
        valhead = torch::jit::load(valhead_path);
        valhead.to(device);
        valhead.eval();
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