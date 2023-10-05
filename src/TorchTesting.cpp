#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  
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
  
  torch::Tensor tensor = torch::zeros({1, 15, 15, 15}, torch::kFloat32).to(torch::kCUDA);
  tensor[0][0] = 1.0f;
  torch::Tensor result = module.forward({tensor}).toTensor().view({225});
  result = torch::softmax(result, -1);

  int index = result.argmax(-1).item<int>();
  std::cout << "Prediction: " << index << std::endl;
  return 0;
}