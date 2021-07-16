cmake_minimum_required(VERSION 3.0.0)
project(computer-vision-dojo
        VERSION 0.0.1
        DESCRIPTION "Computer vision dojo project using c++"
        LANGUAGES CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(computer-vision-dojo main.cpp)

target_link_libraries( computer-vision-dojo ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)