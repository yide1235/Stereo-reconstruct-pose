cmake_minimum_required(VERSION 3.17)
project(TFLiteImageClassification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV Integration
find_package(OpenCV 4.7.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
# add_executable(${PROJECT_NAME} main.cpp)
add_executable(${PROJECT_NAME} main.cpp lib/intersection.cpp detection/detection.cpp movenet/movenet.cpp lib/triangulation.cpp lib/post_processing.cpp)


if(UNIX AND NOT APPLE)
    ADD_LIBRARY(tensorflowlite SHARED IMPORTED)
    set_property(TARGET tensorflowlite PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/libtensorflowlite.so)
    target_link_libraries(${PROJECT_NAME} PRIVATE tensorflowlite ${OpenCV_LIBS})
endif()

