cmake_minimum_required(VERSION 3.20)
project(AlphaGomoku)

message("libtorch path: ${AlphaGomoku_SOURCE_DIR}/libtorch/")
list(APPEND CMAKE_PREFIX_PATH ${AlphaGomoku_SOURCE_DIR}/libtorch/)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 23)
add_executable(AlphaGomoku src/Config.cpp src/Log.cpp src/Style.cpp src/Controller.cpp src/State.cpp src/Node.cpp src/Model.cpp src/Tree.cpp src/Environment.cpp src/Storage.cpp src/Batcher.cpp src/TreeVisualizer.cpp)
target_link_libraries(AlphaGomoku "${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -pedantic")