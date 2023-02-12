#ifndef __CAMERA_ALIGNMENT_HPP__
#define __CAMERA_ALIGNMENT_HPP__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Core>

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h11.h"
}

#define TAG_FAMILY "tag36h11"
#define TAG_SIZE 0.15
#define TAG_ID 3

class camera_alignment {
    public:
        apriltag_detector_t *td;
        apriltag_family_t *tf;

        apriltag_detection_info_t info;

        camera_alignment() {
            this->td = nullptr;
            this->tf = nullptr;
            this->det = nullptr;
        }

        ~camera_alignment() {
            tag36h11_destroy(this->tf);
            apriltag_detector_destroy(this->td);
        }

        void detector_init();

        void set_camera_param(std::vector<double> camera_param);

        bool detect(cv::Mat frame);

        Eigen::Matrix4d pose_estimation();

    private:
        // fx, fy, cx, cy
        std::vector<double> camera_param;

        apriltag_detection_t *det;

        apriltag_pose_t pose;
};

#endif //__CAMERA_ALIGNMENT_HPP__
