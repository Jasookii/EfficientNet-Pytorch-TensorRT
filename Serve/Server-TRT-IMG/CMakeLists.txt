cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(efficientnet)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)


include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
#opencv
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
# torch
# include_directories(/usr/local/include) 
# include_directories(/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include)
# include_directories(/usr/local/lib/python3.6/dist-packages/torch/include)
# link_directories(/usr/local/lib)
# link_directories(/usr/local/lib/python3.6/dist-packages/torch/lib)
# set(CMAKE_PREFIX_PATH /usr/local/lib/python3.6/dist-packages/torch)
# set(Boost_USE_MULTITHREADED ON)
# set(Torch_DIR /usr/local/lib/python3.6/dist-packages/torch)
# find_package(Torch REQUIRED)
# find_package(PythonInterp REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "    OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(efficientnet  ${PROJECT_SOURCE_DIR}/efficientnet.cpp)
# Link your application with OpenCV libraries
# target_link_libraries(efficientnet ${OpenCV_LIBS})
target_link_libraries(efficientnet nvinfer)
target_link_libraries(efficientnet cudart)
target_link_libraries(efficientnet "${OpenCV_LIBRARIES}")
# arget_link_libraries(efficientnet "${TORCH_LIBRARIES}")


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(efficientnet Threads::Threads)

target_compile_options(efficientnet PUBLIC "-pthread")

add_definitions(-O2 -pthread)
set_property(TARGET efficientnet PROPERTY CMAKE_CXX_STANDARD 14)
