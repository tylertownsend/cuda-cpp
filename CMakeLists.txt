############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.5)
project ( prob1 )
find_package(CUDA QUIET REQUIRED)
find_package( OpenCV REQUIRED ) 
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O3 -gencode arch=compute_30,code=sm_30
)
include_directories( ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} )
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (problem_set_1_files main.cpp reference_calc.cpp compare.cpp HW1.cpp)
CUDA_ADD_EXECUTABLE( prob1 ${problem_set_1_files} ${hdr} ${cu})
target_link_libraries( prob1 ${OpenCV_LIBS})
