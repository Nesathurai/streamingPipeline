#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include "open3d/Open3D.h"
#include <kinect_capture.hpp>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <iostream>
#include <multi_kinect_capture.hpp>

using namespace open3d;

constexpr double depth_scale = 1000.0;
constexpr double depth_trunc = 5.0;
constexpr bool convert_rgb_to_intensity = false;
static const core::Device gpu_device("CUDA:0");
static const core::Device cpu_device("CPU:0");

void get_test_image(t::geometry::Image *color, t::geometry::Image *depth, core::Device device);

void create_pano_texture(MultiKinectCapture *multi_cap, uint32_t device_count, cv::Mat *stitched_image);

void get_multiple_test_image(std::vector<t::geometry::Image> *color_img_list, std::vector<t::geometry::Image> *depth_img_list, 
                                core::Device device, int num_devices);

void get_synced_images(MultiKinectCapture *multi_cap, std::vector<t::geometry::Image> *color_img_list, 
                        std::vector<t::geometry::Image> *depth_img_list, 
                        std::vector<cv::Mat> *cv_color_img_list, std::vector<cv::Mat> *cv_depth_img_list);

void icp_all_cloud(std::vector<t::geometry::PointCloud> *cloud_list, std::vector<core::Tensor> *extrinsic_tf_list, 
                    std::vector<core::Tensor> *icp_extrinsic_tf_list);

void transform_all_cloud(std::vector<t::geometry::PointCloud> *cloud_list, std::vector<core::Tensor> *tf_list);

std::vector<core::Tensor> get_camera_tf(MultiKinectCapture *multi_cap);

camera::PinholeCameraIntrinsic get_camera_intrinsic(kinect_capture *cap);

t::geometry::PointCloud rgbd_to_pcl(kinect_capture *cap, camera::PinholeCameraIntrinsic intrinsic);

geometry::AxisAlignedBoundingBox cloud_range_crop(Eigen::Vector3d min_crop_range, Eigen::Vector3d max_crop_range, Eigen::Vector3d box_color);

#endif //__UTIL_HPP__
