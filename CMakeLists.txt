cmake_minimum_required(VERSION 3.28)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 20)

project(MVSLAM2)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(dataset_reader REQUIRED)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp src/*.cc)
file(GLOB_RECURSE headers CONFIGURE_DEPENDS inc/*.h)

add_library(${PROJECT_NAME} SHARED ${srcs} ${headers})
target_include_directories(${PROJECT_NAME} PUBLIC 
    # ${DBoW2_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/inc>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    dataset_reader::dataset_reader
    ceres
)

# Build examples
add_executable(test
./test/test.cc)
target_link_libraries(test ${PROJECT_NAME})

add_executable(triangulate
./test/triangulate.cpp)
target_link_libraries(triangulate ${PROJECT_NAME})