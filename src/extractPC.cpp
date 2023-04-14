#include "open3d/Open3D.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <util.hpp>
#include <texture_mapping.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <camera_alignment.hpp>

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <netdb.h>
#include <inttypes.h>

#include "utils.h"

using namespace open3d;
const char *get_error_text()
{

#if defined(_WIN32)

    static char message[256] = {0};
    FormatMessage(
        FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        0, WSAGetLastError(), 0, message, 256, 0);
    char *nl = strrchr(message, '\n');
    if (nl)
        *nl = 0;
    return message;

#else

    return strerror(errno);

#endif
}

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::system_clock;
using std::chrono::time_point;
high_resolution_clock::time_point start_time = high_resolution_clock::now();
int time_first = 1;
uint32_t device_count = 4;

duration<double, std::milli> delta(std::string msg = "")
{
    // duration<double, std::milli> t2 = (end2 - start2) / 1000;
    duration<double, std::milli> del;
    // time_point<system_clock,duration<double>> zero_{};
    int silent = 0;
    if (msg == "")
    {
        silent = 1;
    }
    if (time_first)
    {
        start_time = high_resolution_clock::now();
        time_first = 0;
        del = (high_resolution_clock::now() - high_resolution_clock::now()) / 1000;
    }
    else
    {
        del = (high_resolution_clock::now() - start_time) / 1000;
        if (!silent)
        {
            std::cout << "  " << msg << ": " << del.count() << " s" << std::endl;
        }
        start_time = high_resolution_clock::now();
    }
    return del;
}

float voxel_size = 0.01; // default 0.01 - optimal is 0.011 in meters
constexpr int block_count = 10000;
constexpr float depth_max = 5.0; // default 5.0
constexpr float trunc_voxel_multiplier = 8.0;

static std::vector<core::Tensor> extrinsic_tf_list;
static std::vector<core::Tensor> intrinsic_list;

int main(int argc, char **argv)
{
    std::string calibStr;
    std::string textStr;
    std::string depthStr;
    std::string nameStr;
    // set parameters for scripting
    
    if (argc == 4)
    {
        calibStr = argv[1];
        textStr = argv[2];
        depthStr = argv[3];
        // string processing to get the name of the file minus .png suffix
        nameStr = textStr.substr(textStr.rfind('/')+1, textStr.size());
        nameStr = nameStr.substr(0, nameStr.rfind('_'));
        // std::cout << calibStr << std::endl;
        // std::cout << textStr << std::endl;
        // std::cout << depthStr << std::endl;
        // std::cout << nameStr << std::endl;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " calib.png texture.png depth.png" << std::endl;
        return -1;
    }

    cv::Mat cv_calib_img;
    cv::Mat cv_color_img;
    cv::Mat cv_depth_img;


    // image stored on disk, use open3d io
    cv_calib_img = cv::imread(calibStr);
    cv_color_img = cv::imread(textStr);
    cv_depth_img = cv::imread(depthStr);
    
    // getting intrinsic matrix from camera calibration file 
    std::string kinect_intrinsic_file = "../camera_calib_files/kinect_000092320412.txt";
    // std::cout << kinect_intrinsic_file << std::endl;

    std::ifstream intrinsic_file(kinect_intrinsic_file);
    if (!intrinsic_file.is_open()) {
        std::cerr << "Could not open the file - '" << kinect_intrinsic_file << "'" << std::endl;
    }

    std::string line;
    std::vector<double> camera_intrinsic;
    while (std::getline(intrinsic_file, line, ' ')) {
        camera_intrinsic.push_back(std::stod(line));
    }

    // get_camera_intrinsic - reading the intrinsic double to another internal storage form
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(1920, 1080, camera_intrinsic.at(0),
                                                                                            camera_intrinsic.at(1),
                                                                                            camera_intrinsic.at(2),
                                                                                            camera_intrinsic.at(3));

    // in start cam, intrinsic is read in 
    core::Tensor intrinsic_t = core::eigen_converter::EigenMatrixToTensor(intrinsic.intrinsic_matrix_);

    // reading camera extrinsic from photo with april tag
    // from util.cpp get_camera_tf(MultiKinectCapture *multi_cap)
    core::Tensor extrinsic_tf;
    
    camera_alignment align;
    align.set_camera_param(camera_intrinsic);
    align.detector_init();
    
    if (align.detect(cv_calib_img)) {
        extrinsic_tf = core::eigen_converter::EigenMatrixToTensor(align.pose_estimation());
    } else {
        std::cout << "Tag " << TAG_ID << " not detected, abort" << std::endl;
        exit(1);
    }


    t::geometry::Image color = core::Tensor(reinterpret_cast<const uint8_t*>(cv_color_img.data),
                                                {cv_color_img.rows,
                                                cv_color_img.cols, 3},
                                                core::UInt8, gpu_device);

    t::geometry::Image depth = core::Tensor(reinterpret_cast<const uint16_t*>(cv_depth_img.data),
                                                    {cv_depth_img.rows,
                                                    cv_depth_img.cols, 1},
                                                    core::UInt16, gpu_device);

    // std::shared_ptr<geometry::RGBDImage> rgbd_image_ptr = geometry::RGBDImage::CreateFromColorAndDepth(color.ToLegacy(), depth.ToLegacy(), 
    //                                                 depth_scale, 
    //                                                 depth_trunc,
    //                                                 convert_rgb_to_intensity);
    
    std::shared_ptr<geometry::RGBDImage> rgbd_image_ptr = geometry::RGBDImage::CreateFromColorAndDepth(color.ToLegacy(), depth.ToLegacy(), 
                                                    depth_scale, 
                                                    depth_trunc,
                                                    convert_rgb_to_intensity);

    geometry::RGBDImage *rgbd_image = rgbd_image_ptr.get();

    std::shared_ptr<geometry::PointCloud> cloud = geometry::PointCloud::CreateFromRGBDImage(*rgbd_image, intrinsic);
    t::geometry::PointCloud cloud_t = t::geometry::PointCloud::FromLegacy(*(cloud.get()));
    t::geometry::PointCloud cloud_tf = cloud_t.Transform(extrinsic_tf.Inverse());

    char outPC[1024] = {0};
    sprintf(outPC, "/home/sc/streamingPipeline/analysisData/meshPCs/%s.ply", nameStr.c_str());
    open3d::t::io::WritePointCloud(outPC, cloud_tf);

    return 1;
}
