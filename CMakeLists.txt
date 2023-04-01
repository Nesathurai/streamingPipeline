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

add_executable(extractMesh
    src/extractMesh.cpp
    src/kinect_capture.cpp
    src/camera_alignment.cpp
    src/multi_kinect_capture.cpp
    src/util.cpp
    src/texture_mapping.cpp
    include/cv_convert_util.hpp
)

add_executable(extractPC
    src/extractPC.cpp
    src/kinect_capture.cpp
    src/camera_alignment.cpp
    src/multi_kinect_capture.cpp
    src/util.cpp
    src/texture_mapping.cpp
    include/cv_convert_util.hpp
)

add_executable(mergePC
    src/mergePC.cpp
    src/kinect_capture.cpp
    src/camera_alignment.cpp
    src/multi_kinect_capture.cpp
    src/util.cpp
    src/texture_mapping.cpp
    include/cv_convert_util.hpp
)

add_executable(mergeMesh
    src/mergeMesh.cpp
    src/kinect_capture.cpp
    src/camera_alignment.cpp
    src/multi_kinect_capture.cpp
    src/util.cpp
    src/texture_mapping.cpp
    include/cv_convert_util.hpp
)


target_link_libraries(reconstruction -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D
                        apriltag::apriltag
                        k4a::k4a
                        ${OpenCV_LIBS})
                        
target_link_libraries(client -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D)

target_link_libraries(extractMesh -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D
                        apriltag::apriltag
                        k4a::k4a
                        ${OpenCV_LIBS})

target_link_libraries(extractPC -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D
                        apriltag::apriltag
                        k4a::k4a
                        ${OpenCV_LIBS})

                        
target_link_libraries(mergePC -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D
                        apriltag::apriltag
                        k4a::k4a
                        ${OpenCV_LIBS})


target_link_libraries(mergeMesh -Wl,--whole-archive draco -Wl,--no-whole-archive
                        Open3D::Open3D
                        apriltag::apriltag
                        k4a::k4a
                        ${OpenCV_LIBS})

target_compile_options(client PUBLIC
  "-O3"
  )

target_compile_options(extractMesh PUBLIC
  "-O3"
  )
target_compile_options(extractPC PUBLIC
  "-O3"
  )
target_compile_options(mergePC PUBLIC
  "-O3"
  )
target_compile_options(mergeMesh PUBLIC
  "-O3"
  )
target_compile_options(reconstruction PUBLIC
  "-ggdb"
  "-O3"
  )