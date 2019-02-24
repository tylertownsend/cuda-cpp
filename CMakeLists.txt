############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.5)
project ( prob1 LANGUAGES CUDA CXX)

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
set(problem_set_1_files main.cpp reference_calc.cpp compare.cpp)

add_executable(prob1 ${problem_set_1_files} ${hdr} ${cu})
target_include_directories(prob1 PUBLIC "${CMAKE_SOURCE_DIR}/*h")
target_link_libraries( prob1 ${OpenCV_LIBS})
target_link_libraries( prob1 ${CUDA_LIBS})
