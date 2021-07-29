cmake_minimum_required(VERSION 3.0.0)
project(cv_challenge_1
        VERSION 0.0.1
        DESCRIPTION "First computer vision challenge"
        LANGUAGES CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cv_challenge_one main.cpp)

target_link_libraries( cv_challenge_one ${OpenCV_LIBS} )

target_compile_features(cv_challenge_one PRIVATE cxx_std_20)

target_include_directories(cv_challenge_one PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)