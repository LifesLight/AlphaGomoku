#include "Model.h"

Model::Model(std::string model_path)
    : Model(model_path, torch::kCPU)
{   }

Model::Model(std::string model_path, torch::Device device)
    : Model(model_path, device, "unnamed")
{   }

Model::Model(torch::jit::script::Module  jit_module, std::string name)
    : Model(jit_module, torch::kCPU, name)
{   }

Model::Model(std::string model_path, torch::Device device, std::string name)
    : device(device), model_name(name)
{
    try
    {
        neural_network = torch::jit::load(model_path);
        neural_network.to(device);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load model from:" << std::endl << model_path << std::endl;
    }
}


Model::Model(torch::jit::script::Module jit_module)
    : Model(jit_module, torch::kCPU)
{   }

Model::Model(torch::jit::script::Module jit_module, torch::Device device)
    : Model(jit_module, device, "unnamed")
{   }

Model::Model(std::string model_path, std::string name)
    : Model(model_path, torch::kCPU, name)
{   }

Model::Model(torch::jit::script::Module jit_module, torch::Device device, std::string name)
    : device(device), model_name(name)
{
    neural_network = jit_module;
    neural_network.to(device);
}


std::tuple<torch::Tensor, torch::Tensor> Model::forward(torch::Tensor input)
{
    input = input.to(device);
    auto result = neural_network.forward({input});
    std::vector<at::IValue> output_tuple = result.toTuple()->elements();

    // Extract policy and value outputs
    torch::Tensor policy_output = output_tuple[0].toTensor();
    torch::Tensor value_output = output_tuple[1].toTensor();
    policy_output = torch::softmax(policy_output, -1);

    return std::tuple<torch::Tensor, torch::Tensor>(policy_output, value_output);
}

std::string Model::getName()
{
    return model_name;
}