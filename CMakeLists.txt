cmake_minimum_required(VERSION 3.10)

project(dipper LANGUAGES CXX)
include(CheckLanguage)

# ------------------ Versioning ------------------
set(dipper_VERSION "?.?.?")
execute_process(COMMAND head -n1 ${CMAKE_SOURCE_DIR}/version.txt OUTPUT_VARIABLE VERSION)
string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH "${VERSION}")
if("${VERSION_PATCH}" MATCHES "[0-9]+")
    set(dipper_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
endif()
configure_file(src/version.hpp.in ${CMAKE_SOURCE_DIR}/src/version.hpp)

# For CUDA
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA compiler detected at: ${CMAKE_CUDA_COMPILER}")
        TRY_RUN(
                FIND_GPU_RESULT
                FIND_GPU_COMPILE_RESULT
                ${CMAKE_CURRENT_BINARY_DIR}/
                ${CMAKE_CURRENT_BINARY_DIR}/../src/findgpu.cu
                RUN_OUTPUT_VARIABLE GPU_ARCH_OUTPUT
        )
        if("${FIND_GPU_COMPILE_RESULT}" AND ("${FIND_GPU_RESULT}" EQUAL 0))
                message(STATUS "CUDA-capable GPU found. GPU architecture: ${GPU_ARCH_OUTPUT}")
                if(${CMAKE_VERSION} VERSION_LESS "3.18")
                        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${GPU_ARCH_OUTPUT}")
                else()
                        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH_OUTPUT})
                endif()
        else()
                message(STATUS "No CUDA-capable GPU found. Generating binary to support multiple GPU generations.")
                set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)
        endif()
else()
        message(FATAL_ERROR "No CUDA found")
endif()

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# find_package(CUDA 10.0 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4 -Wall -g")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -rdc=true --extended-lambda -Xcompiler -std=c++17")

find_package(Boost COMPONENTS program_options REQUIRED)

find_package(ZLIB REQUIRED)
find_path(ZLIB_INCLUDE_DIR zlib.h)
include_directories(${ZLIB_INCLUDE_DIR})

# include(${TBB_DIR}/cmake/TBBBuild.cmake)
# tbb_build(TBB_ROOT ${TBB_DIR} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)
# find_package(TBB REQUIRED tbbmalloc tbbmalloc_proxy tbb_preview)

add_executable (dipper
        src/tree_generation.cu
        src/placement.cu
        src/mash.cu
        src/twoBitCompressor.cpp
        src/fourBitCompressor.cpp
        src/matrix_reader.cu
        src/MSA.cu
        src/tree.cpp
        src/neighborJoining.cu
        src/placement_close_k.cu
        src/divide_and_conquer/mash.cu
        src/divide_and_conquer/mash.cpp
        src/divide_and_conquer/msa.cu
        src/divide_and_conquer/placement_close_k.cpp
        src/divide_and_conquer/placement_close_k.cu)

add_executable (dipper_cpu
        src/tree_generation.cu
        src/placement.cu
        cpu/mash.cu
        src/twoBitCompressor.cpp
        src/fourBitCompressor.cpp
        cpu/matrix_reader.cu
        src/MSA.cu
        src/tree.cpp
        src/neighborJoining.cu
        cpu/placement_close_k.cu
        src/divide_and_conquer/mash.cu
        src/divide_and_conquer/mash.cpp
        src/divide_and_conquer/msa.cu
        src/divide_and_conquer/placement_close_k.cpp
        src/divide_and_conquer/placement_close_k.cu)


find_package(TBB REQUIRED CONFIG)
set_target_properties(dipper PROPERTIES CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
set_target_properties(dipper_cpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
find_library(STDCXXFS NAMES stdc++fs)
if (STDCXXFS)
  target_link_libraries(dipper PRIVATE ${STDCXXFS})
  target_link_libraries(dipper_cpu PRIVATE ${STDCXXFS})
endif()
target_link_libraries(dipper PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} TBB::tbb)
target_link_libraries(dipper_cpu PRIVATE ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} TBB::tbb)

install(TARGETS dipper
        RUNTIME DESTINATION bin)

install(TARGETS dipper_cpu
        RUNTIME DESTINATION bin)
