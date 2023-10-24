cmake_minimum_required(VERSION 3.20)
project(AlphaGomoku)

message("libtorch path: ${AlphaGomoku_SOURCE_DIR}/libtorch/")
list(APPEND CMAKE_PREFIX_PATH ${AlphaGomoku_SOURCE_DIR}/libtorch/)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 23)
add_executable(AlphaGomoku src/Controller.cpp src/State.cpp src/Node.cpp src/Gamestate.cpp src/Model.cpp src/Tree.cpp src/Environment.cpp src/Batcher.cpp)
target_link_libraries(AlphaGomoku "${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "-O3")