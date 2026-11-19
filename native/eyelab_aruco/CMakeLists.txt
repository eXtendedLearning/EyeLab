# EyeLab ArUco plugin for Android arm64 (B3a). Built by tools/build_aruco_plugin.sh with the NDK that ships
# with Unity, against the static libraries of the OpenCV Android SDK (vendor/opencv-android/, gitignored), so
# the plugin is one self-contained .so with no libopencv_java4.so beside it.
cmake_minimum_required(VERSION 3.22)
project(eyelab_aruco CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d objdetect)

add_library(eyelab_aruco SHARED eyelab_aruco.cpp)
target_include_directories(eyelab_aruco PRIVATE ${OpenCV_INCLUDE_DIRS})
# c++_static by name: with Unity under "Program Files" CMake calls clang++ by its 8.3 short name (CLANG_~1.EXE),
# clang then runs in C mode and -static-libstdc++ links no C++ runtime at all.
target_link_libraries(eyelab_aruco PRIVATE ${OpenCV_LIBS} log c++_static c++abi)
target_compile_options(eyelab_aruco PRIVATE -O2 -ffunction-sections -fdata-sections)
# 16 KB pages: required by Android 15+ devices that use them (portability, ADR-0004).
target_link_options(eyelab_aruco PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384 -s)
