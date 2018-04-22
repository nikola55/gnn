cmake_minimum_required(VERSION 2.8)

project(gnn)

set(SRC_LIST
./src/blas_matrix_oper.cpp
./src/neural_net.cpp
)

include_directories(include)

add_library(gnn SHARED ${SRC_LIST})

add_executable(test test.cpp)

target_link_libraries(test gnn gsl gslcblas m)