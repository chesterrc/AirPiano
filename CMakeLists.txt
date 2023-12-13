cmake_minimum_required(VERSION 3.0.0)
project(AirPiano)

include(CTest)
enable_testing()

SET(OpenCV_DIR C:/Users/livti/Downloads/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )


