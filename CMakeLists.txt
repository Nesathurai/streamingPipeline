set(CMAKE_INSTALL_PREFIX open3d_install_0.16.0)

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(reconstruction)

find_package(Open3D 0.16.0 REQUIRED)
find_package(OpenCV REQUIRED  core imgproc imgcodecs highgui)
find_package(apriltag REQUIRED)
find_package(k4a REQUIRED)
find_package(draco REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)
#include_directories(/home/sc/Open3D-0.16.0 include)

add_executable(reconstruction
    src/reconstruction.cpp
    src/kinect_capture.cpp
    src/camera_alignment.cpp
    src/multi_kinect_capture.cpp
    src/util.cpp
    src/texture_mapping.cpp
    include/cv_convert_util.hpp
)

add_executable(client
    src/client.cpp
)

target_link_libraries(reconstruction -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D
                        apriltag::apriltag
                        k4a::k4a
                        ${OpenCV_LIBS})

target_link_libraries(client -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D)

target_compile_options(client PUBLIC
  #"-Werror" # Do not comment this out after Project 8!
  "-ggdb"
  "-O3"
  #"-Wunreachable-code"
  #"-g"
  #"-Wextra"
  #"-Wall"
  #"-Wfloat-equal"
  #"-Wconversion"
  #"-Wparentheses"
  #"-pedantic"
  #"-Wunused-parameter"
  #"-Wunused-variable"
  #"-Wreturn-type"
  #"-Wunused-function"
  #"-Wredundant-decls"
  #"-Wreturn-type"
  #"-Wredundant-decls"
  #"-Wunused-value"
  #"-Wswitch-default"
  #"-Wuninitialized" "-Winit-self"
  )

  target_compile_options(reconstruction PUBLIC
  #"-Werror" # Do not comment this out after Project 8!
  "-ggdb"
  "-O3"
  #"-Wunreachable-code"
  #"-g"
  #"-Wextra"
  #"-Wall"
  #"-Wfloat-equal"
  #"-Wconversion"
  #"-Wparentheses"
  #"-pedantic"
  #"-Wunused-parameter"
  #"-Wunused-variable"
  #"-Wreturn-type"
  #"-Wunused-function"
  #"-Wredundant-decls"
  #"-Wreturn-type"
  #"-Wredundant-decls"
  #"-Wunused-value"
  #"-Wswitch-default"
  #"-Wuninitialized" "-Winit-self"
  )