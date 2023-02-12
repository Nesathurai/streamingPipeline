#ifndef __KINECT_CAPTURE_HPP__
#define __KINECT_CAPTURE_HPP__

#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <chrono>

#define VISUALIZE_POINT_SIZE 3

class kinect_capture {
    public:
        k4a::device device;
        
        k4a::image colorImage;
        k4a::image depthImage;
        k4a::image transformedDepthImage;
        k4a::image xyzImage;

        cv::Mat cv_color_img;
        cv::Mat cv_depth_img;

        std::vector<double> camera_intrinsic;
        Eigen::Matrix4f pose;

        std::string serial_num;
        k4a_device_configuration_t config;

        kinect_capture(uint32_t device_idx) {
            this->device = k4a::device::open(device_idx);
            this->serial_num = this->device.get_serialnum();
            
            this->config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
            this->config.camera_fps = K4A_FRAMES_PER_SECOND_15;
            this->config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
            this->config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
            this->config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
            // this->config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
            this->config.synchronized_images_only = true;
            
            // this->device.start_cameras(&this->config);

            this->calibration = this->device.get_calibration(this->config.depth_mode, this->config.color_resolution);

            // std::cout << this->calibration.color_camera_calibration.intrinsics.parameters.param.fx << std::endl;
            // std::cout << this->calibration.color_camera_calibration.intrinsics.parameters.param.fy << std::endl;
            // std::cout << this->calibration.color_camera_calibration.intrinsics.parameters.param.cx << std::endl;
            // std::cout << this->calibration.color_camera_calibration.intrinsics.parameters.param.cy << std::endl;

            this->transformation = k4a::transformation(this->calibration);

            this->pose = Eigen::Matrix4f::Identity(4, 4);

            this->calibration_type = K4A_CALIBRATION_TYPE_COLOR;
        }

        ~kinect_capture() {}

        void get_intrinsic_calibration();

        void capture_frame();

        void to_cv_img();

        void tf_depth_image();

        void depth_to_pcl();

    private:
        k4a::calibration calibration;
        k4a::capture capture;
        k4a::transformation transformation;
        k4a_calibration_type_t calibration_type;

        int color_width;
        int color_height;
};

#endif //__KINECT_CAPTURE_HPP__
