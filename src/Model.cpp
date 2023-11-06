#include "Model.h"

torch::jit::script::Module load_model(std::string path)
{
    // Always load on CPU
    torch::jit::script::Module model = torch::jit::load(path, torch::kCPU);
    model.to(TorchDefaultDevice);
    model.to(TorchDefaultScalar);
    model.eval();  
    return model;  
}

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path)
    : Model(resnet_path, polhead_path, valhead_path, DefaultSimulations)
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, int simulations)
    : Model(resnet_path, polhead_path, valhead_path, simulations, "unnamed")
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, std::string name)
    : Model(resnet_path, polhead_path, valhead_path, DefaultSimulations, name)
{   }

Model::Model(std::string resnet_path, std::string polhead_path, std::string valhead_path, int simulations, std::string name)
    : model_name(name), simulations(simulations), device(TorchDefaultDevice), dtype(TorchDefaultScalar)
{
    // Load resnet
    try
    {
        resnet = load_model(resnet_path);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load resnet from:" << std::endl << resnet_path << std::endl;
    }

    // Load polhead
    try
    {
        polhead = load_model(polhead_path);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load policyhead from:" << std::endl << polhead_path << std::endl;
    }

    // Load valhead
    try
    {
        valhead = load_model(valhead_path);
    }
    catch (const c10::Error& e)
    {
        std::cout << "[Model][E]: Could not load valuehead from:" << std::endl << valhead_path << std::endl;
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
    torch::Tensor policy_output = policy_result.toTensor().detach();
    torch::Tensor value_output = value_result.toTensor().detach();
    policy_output = torch::softmax(policy_output, -1);
    policy_output = policy_output.to(torch::kCPU);
    value_output = value_output.to(torch::kCPU);

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
    return autoloadModel(name, DefaultSimulations);
}

Model* Model::autoloadModel(std::string name, int simulations)
{
    if (Utils::checkEnv("LOGGING", "INFO"))
        std::cout << "[Model][I]: Autoloading: " << name << std::endl;

    std::string general_path = ModelPath;
    std::string resnet_path = general_path + "ResNet/" + name;
    std::string policy_path = general_path + "PolHead/" + name;
    std::string value_path = general_path + "ValHead/" + name;

    Model* loaded_model;
    try
    {
        loaded_model = new Model(resnet_path, policy_path, value_path, simulations, name);
    }
    catch(const std::exception& e)
    {
        std::cerr << "[Model][E]: Failed to autoload Model" << '\n' << e.what() << '\n';
    }
    return loaded_model;
}