#include <kinect_capture.hpp>
#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cv_convert_util.hpp>

using namespace std;

void kinect_capture::get_intrinsic_calibration() {
    std::string kinect_intrinsic_file = "../camera_calib_files/kinect_" + this->serial_num + ".txt";
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
    std::cout << camera_intrinsic.at(0) << ", " << camera_intrinsic.at(1) 
                << ", " << camera_intrinsic.at(2) << ", " << camera_intrinsic.at(3) << std::endl;
    
    this->camera_intrinsic = camera_intrinsic;
}

void kinect_capture::capture_frame() {
    this->device.get_capture(&this->capture, std::chrono::milliseconds{K4A_WAIT_INFINITE});

    this->depthImage = this->capture.get_depth_image();
    this->colorImage = this->capture.get_color_image();

    while (!this->colorImage.is_valid() || !this->depthImage.is_valid()) {
        this->device.get_capture(&this->capture, std::chrono::milliseconds{K4A_WAIT_INFINITE});
        this->colorImage = this->capture.get_color_image();
        this->depthImage = this->capture.get_depth_image();
    }

    // while (!this->depthImage.is_valid()) {
    //     this->device.get_capture(&this->capture, std::chrono::milliseconds{K4A_WAIT_INFINITE});
    //     this->depthImage = this->capture.get_depth_image();
    // }
    
    cv::Mat cv_color = k4a::get_mat(this->colorImage);
    this->cv_color_img = cv_color;

    tf_depth_image();
    to_cv_img();
    // cv::Mat cv_depth = k4a::get_mat(this->transformedDepthImage);
    // this->cv_depth_img = cv_depth;
}

void kinect_capture::to_cv_img() {
    uint8_t* buffer = this->transformedDepthImage.get_buffer();
    int rows = this->transformedDepthImage.get_height_pixels();
    int cols = this->transformedDepthImage.get_width_pixels();
    this->cv_depth_img = cv::Mat(rows , cols, CV_16U, (void*)buffer, cv::Mat::AUTO_STEP);

    // uint8_t* buffer = this->colorImage.get_buffer();
    // int rows = this->colorImage.get_height_pixels();
    // int cols = this->colorImage.get_width_pixels();
    // this->cv_color_img = cv::Mat(rows , cols, CV_8UC4, (void*)buffer, cv::Mat::AUTO_STEP);

    // BGRA to RGB
    // cv::cvtColor(this->cv_color_img, this->cv_color_img, cv::COLOR_BGRA2RGB);
}

void kinect_capture::tf_depth_image() {
    this->transformedDepthImage = k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16,
                                                        this->colorImage.get_width_pixels(),
                                                        this->colorImage.get_height_pixels(),
                                                        this->colorImage.get_width_pixels() * (int)sizeof(uint16_t));

    this->transformedDepthImage = this->transformation.depth_image_to_color_camera(this->depthImage);
}

void kinect_capture::depth_to_pcl() {
    this->xyzImage = k4a::image::create(K4A_IMAGE_FORMAT_CUSTOM,
                                        this->transformedDepthImage.get_width_pixels(),
                                        this->transformedDepthImage.get_height_pixels(),
                                        this->transformedDepthImage.get_width_pixels() * 3 * (int)sizeof(int16_t));
    
    this->xyzImage = this->transformation.depth_image_to_point_cloud(this->transformedDepthImage,
                                                            this->calibration_type);
}
