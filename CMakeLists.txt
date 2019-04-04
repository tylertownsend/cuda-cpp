cmake_minimum_required(VERSION 3.5)
project ( prob3 LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
find_package( OpenCV REQUIRED ) 

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O3 -gencode arch=compute_30,code=sm_30
)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDA_INCLUDE_DIRS} )
set(problem_set_3_files main.cpp reference_calc.cpp compare.cpp loadSaveImage.cpp)

add_executable(prob3 ${problem_set_3_files} ${hdr} ${cu})
target_include_directories(prob3 PUBLIC "${CMAKE_SOURCE_DIR}/*h")
target_link_libraries( prob3 ${OpenCV_LIBS})
target_link_libraries( prob3 ${CUDA_LIBS})
