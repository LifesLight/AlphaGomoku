#include "Model.h"

Model::Model(torch::jit::script::Module jit_module, torch::Device device)
    : device(device)
{
    neural_network = jit_module;
    neural_network.to(device);
}

Model::Model(std::string model_path, torch::Device device)
    : device(device)
{
    try 
    {
        neural_network = torch::jit::load(model_path);
        neural_network.to(device);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error loading the model\n" << e.what() << std::endl;
    }
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