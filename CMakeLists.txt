cmake_minimum_required(VERSION 3.20)

# 必须在 project() 之前设置：project() 会用编译器的默认架构初始化这个变量。
# 默认 RTX 4090（sm_89）；其他卡用 -DCMAKE_CUDA_ARCHITECTURES=80 等覆盖。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")
endif()

project(cuda_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# -lineinfo 让 Nsight Compute 能把指标对应到源码行；ptxas -v 打印寄存器与共享内存用量。
set(LAB_CUDA_FLAGS -O3 -lineinfo --ptxas-options=-v)

function(add_bench name)
  add_executable(${name} src/${name}.cu)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${LAB_CUDA_FLAGS}>)
  target_link_libraries(${name} PRIVATE CUDA::cudart CUDA::cublas)
endfunction()

add_bench(sgemm)
add_bench(hgemm_wmma)
add_bench(reduce)
add_bench(softmax)
add_bench(rmsnorm)
add_bench(transpose)
