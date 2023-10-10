#include "Model.h"


Model::Model(torch::jit::script::Module jit_module, torch::Device device)
{
    neural_network = jit_module;
}

Model::Model(std::string model_path, torch::Device device)
    : device(device)
{
    try 
    {
        neural_network = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "Error loading the model\n" << e.what() << std::endl;
    }
}

std::tuple<torch::Tensor, torch::Tensor> Model::forward(torch::Tensor input)
{
    //input.to(device);
    std::cout << "Inside forward" << std::endl;
    input = torch::zeros({1, 9, 15, 15}, torch::kFloat32);
    auto result = neural_network.forward({input});
    std::cout << "Forwarded" << std::endl;
    std::vector<at::IValue> output_tuple = result.toTuple()->elements();

    // Extract policy and value outputs
    torch::Tensor policy_output = output_tuple[0].toTensor();
    torch::Tensor value_output = output_tuple[1].toTensor();
    policy_output = torch::softmax(policy_output, -1);
    std::cout << "Softmaxed" << std::endl;
        
    return std::tuple<torch::Tensor, torch::Tensor>(policy_output, value_output);
}