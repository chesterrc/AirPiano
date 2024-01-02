cmake_minimum_required(VERSION 3.0.0)
project(AirPiano VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set("OpenCV_DIR" C:/Users/livti/Desktop/Projects/MLStuff/opencv/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)



