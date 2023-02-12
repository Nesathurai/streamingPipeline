#include <camera_alignment.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Core>

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag36h11.h"
}

void camera_alignment::detector_init() {
    this->tf = tag36h11_create();
    this->td = apriltag_detector_create();
    this->td->nthreads = 8;
    apriltag_detector_add_family(td, tf);

    this->info.det = this->det;
    this->info.tagsize = TAG_SIZE;
    this->info.fx = this->camera_param.at(0);
    this->info.fy = this->camera_param.at(1);
    this->info.cx = this->camera_param.at(2);
    this->info.cy = this->camera_param.at(3);
}

void camera_alignment::set_camera_param(std::vector<double> camera_param) {
    this->camera_param = camera_param;
}

bool camera_alignment::detect(cv::Mat frame) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    image_u8_t im = { .width = gray.cols,
                        .height = gray.rows,
                        .stride = gray.cols,
                        .buf = gray.data };

    zarray_t *detections = apriltag_detector_detect(td, &im);
    int num_detection = zarray_size(detections);
    std::cout << "number tags detected: " << num_detection << std::endl;

    // if (num_detection >= 1) {
    //     zarray_get(detections, 0, &(this->info.det));
    //     std::cout << "selected tag id: " << this->info.det->id << std::endl;
    // }
    for (int i = 0; i < num_detection; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        std::cout << "current tag id: " << det->id << std::endl;
        if (det->id == TAG_ID) {
            this->info.det = det;
            std::cout << "selected tag id: " << this->info.det->id << std::endl;
            return true;
        }
    }
    

    return false;
}

Eigen::Matrix4d camera_alignment::pose_estimation() {
    double err = estimate_tag_pose(&(this->info), &(this->pose));

    Eigen::Matrix4d tf = Eigen::Matrix4d::Zero();

    int num_rows = this->pose.R->nrows;
    int num_cols = this->pose.R->ncols;
    // Rotation
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            int index = row * num_cols + col;
            tf(row, col) = this->pose.R->data[index];
        }
    }
    // Translation
    num_rows = this->pose.t->nrows;
    num_cols = this->pose.t->ncols;
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            int index = row * num_cols + col;
            tf(row, 3) = this->pose.t->data[index];
        }
    }
    tf(3, 3) = 1;

    return tf;
}
