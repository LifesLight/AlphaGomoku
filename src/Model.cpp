#include "Model.h"

Model::Model(torch::jit::script::Module  jit_module)
{
    neural_network = jit_module;
    neural_network.to(torch::kCUDA);
}

Model::Model(std::string model_path)
{
    try 
    {
        neural_network = torch::jit::load(model_path);
        neural_network.to(torch::kCUDA);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error loading the model\n" << e.what() << std::endl;
    }
}

std::tuple<torch::Tensor, float> Model::forward(torch::Tensor input)
{
    input = input.to(torch::kCUDA);
    auto result = neural_network.forward({input});
    std::vector<at::IValue> output_tuple = result.toTuple()->elements();

    // Extract policy and value outputs
    torch::Tensor policy_output = output_tuple[0].toTensor();
    torch::Tensor value_output = output_tuple[1].toTensor();

    policy_output = torch::softmax(policy_output, -1);
    float eval = value_output.item<float>();

    return std::tuple<torch::Tensor, float>(policy_output, eval);
}