cmake_minimum_required(VERSION 3.20)
project(AlphaGomoku)

file(   DOWNLOAD https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
        ${AlphaGomoku_SOURCE_DIR}/libtorch.zip
        SHOW_PROGRESS)
message("Extracting...")
file(   ARCHIVE_EXTRACT INPUT
        ${AlphaGomoku_SOURCE_DIR}/libtorch.zip
	DESTINATION ${AlphaGomoku_SOURCE_DIR}/)
message("Library path: ${AlphaGomoku_SOURCE_DIR}/libtorch/")

list(APPEND CMAKE_PREFIX_PATH ${AlphaGomoku_SOURCE_DIR}/libtorch/)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 23)
add_executable(AlphaGomoku src/MCTS.cpp src/State.cpp src/Node.cpp src/Config.cpp)
target_link_libraries(AlphaGomoku "${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "-O3")
