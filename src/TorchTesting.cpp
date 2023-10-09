#include "Includes.h"
#include "Model.h"
#include "Gamestate.h"
#include "Node.h"

int main(int argc, const char* argv[]) {
  
  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;


    Model* network = new Model(argv[1]);
  
    Node* root = new Node(network);
    // TODO Jakob macht das morgen :)

    Gamestate jimmy = Gamestate(root);
    std::cout << jimmy.sliceToString(0) << std::endl;
}
