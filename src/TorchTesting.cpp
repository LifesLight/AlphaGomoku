#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  
  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;

  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
    module.to(torch::kCUDA);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n" << e.what() << std::endl;
    return -1;
  }

  std::cout << "Loaded Model.\n";
  
  torch::Tensor tensor = torch::zeros({1, 11, 15, 15}, torch::kFloat32).to(torch::kCUDA);
  tensor[0][0] = 0.0f;
  torch::Tensor result = module.forward({tensor}).toTensor().view({225});
  result = torch::softmax(result, -1);

  int index = result.argmax(-1).item<int>();
  std::cout << "Prediction: " << index << "(" << index / 15 << "," << index % 15 << ")" << std::endl;
  return 0;
}