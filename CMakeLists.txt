cmake_minimum_required(VERSION 3.0.0)
project(AirPiano VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

SET(OpenCV_DIR C:/Users/livti/Downloads/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(AirPiano main.cpp)

target_link_libraries( AirPiano ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
