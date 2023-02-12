#include "open3d/Open3D.h"
#include <kinect_capture.hpp>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <iostream>
#include <util.hpp>
#include <Eigen/Core>
#include <camera_alignment.hpp>
#include <multi_kinect_capture.hpp>

using namespace open3d;

void get_test_image(t::geometry::Image *color, t::geometry::Image *depth, core::Device device) {
    std::string img_path = "../test_images/whiteboard/";

    *depth = (*t::io::CreateImageFromFile(img_path + "0_depth.png"));
    *color = (*t::io::CreateImageFromFile(img_path + "0_rgb.png"));

    *depth = depth->To(device);
    *color = color->To(device);
}

void create_pano_texture(MultiKinectCapture *multi_cap, uint32_t device_count, cv::Mat *stitched_image) {
    std::vector<cv::Mat> cam_images(device_count);
    for (uint32_t i = 0; i < device_count; i++) {
        multi_cap->get_synchronized_captures();
        cam_images.at(i) = multi_cap->capture_devices.at(i)->cv_color_img;
    }
    cv::hconcat(cam_images, *stitched_image);
    cv::cvtColor(*stitched_image, *stitched_image, cv::COLOR_BGR2RGB);
    cv::imwrite("stitched_image.jpg", *stitched_image);
}

void get_multiple_test_image(std::vector<t::geometry::Image> *color_img_list, std::vector<t::geometry::Image> *depth_img_list, 
                                core::Device device, int num_devices) {
    // std::string img_path = "../test_images/multi_cap/";
    std::string img_path = "../2101_capture/";

    for (int i = 0; i < num_devices; i++) {
        color_img_list->push_back(*t::io::CreateImageFromFile(img_path + "color_image_" + std::to_string(i) + ".jpg"));
        depth_img_list->push_back(*t::io::CreateImageFromFile(img_path + "depth_image_" + std::to_string(i) + ".png"));
        color_img_list->at(i) = color_img_list->at(i).To(device);
        depth_img_list->at(i) = depth_img_list->at(i).To(device);
    }
}

void get_synced_images(MultiKinectCapture *multi_cap, std::vector<t::geometry::Image> *color_img_list, 
                        std::vector<t::geometry::Image> *depth_img_list, 
                        std::vector<cv::Mat> *cv_color_img_list, std::vector<cv::Mat> *cv_depth_img_list) {
    multi_cap->get_synchronized_captures();
    
    for (int i = 0; i < multi_cap->get_num_devices(); i++) {
        kinect_capture *cap = multi_cap->capture_devices.at(i);
        t::geometry::Image color = core::Tensor(reinterpret_cast<const uint8_t*>(cap->cv_color_img.data),
                                                    {cap->cv_color_img.rows,
                                                    cap->cv_color_img.cols, 3},
                                                    core::UInt8, gpu_device);

        t::geometry::Image depth = core::Tensor(reinterpret_cast<const uint16_t*>(cap->cv_depth_img.data),
                                                        {cap->cv_depth_img.rows,
                                                        cap->cv_depth_img.cols, 1},
                                                        core::UInt16, gpu_device);
    
        color_img_list->at(i) = color;
        depth_img_list->at(i) = depth;
        cv_color_img_list->at(i) = cap->cv_color_img;
        cv_depth_img_list->at(i) = cap->cv_depth_img;
    }
}

void transform_all_cloud(std::vector<t::geometry::PointCloud> *cloud_list, std::vector<core::Tensor> *tf_list) {
    for (uint32_t i = 0; i < cloud_list->size(); i++) {
        t::geometry::PointCloud cloud = cloud_list->at(i);
        t::geometry::PointCloud cloud_tf = cloud.Transform(tf_list->at(i).Inverse());
        cloud_list->at(i) = cloud_tf;
        // open3d::io::WritePointCloud("cloud_" + std::to_string(i) + ".pcd", cloud_list.at(i).ToLegacy());
    }
}

std::vector<core::Tensor> get_camera_tf(MultiKinectCapture *multi_cap) {
    std::vector<core::Tensor> tf_list;
    for (uint32_t dev_idx = 0; dev_idx < multi_cap->get_num_devices(); dev_idx++) {
        camera_alignment align;
        kinect_capture *capture_device = multi_cap->capture_devices.at(dev_idx);
        align.set_camera_param(capture_device->camera_intrinsic);
        align.detector_init();
        
        if (align.detect(capture_device->cv_color_img)) {
            core::Tensor tf = core::eigen_converter::EigenMatrixToTensor(align.pose_estimation());
            tf_list.push_back(tf);
        } else {
            std::cout << "Tag " << TAG_ID << " not detected, abort" << std::endl;
            exit(1);
        }
    }

    // // read extrinsic numpy txt files
    // for (uint32_t dev_idx = 0; dev_idx < multi_cap->get_num_devices(); dev_idx++) {
    //     kinect_capture *capture_device = multi_cap->capture_devices.at(dev_idx);
    //     std::string serial_num = capture_device->serial_num;
    //     core::Tensor extrinsic = t::io::ReadNpy("/home/taojin/ws/camera_calibration/camera_apriltag/" + serial_num + "/extrinsic.npy");
    //     tf_list.push_back(extrinsic);
    // }

    return tf_list;
}

camera::PinholeCameraIntrinsic get_camera_intrinsic(kinect_capture *cap) {
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(1920, 1080, cap->camera_intrinsic.at(0),
                                                                                            cap->camera_intrinsic.at(1),
                                                                                            cap->camera_intrinsic.at(2),
                                                                                            cap->camera_intrinsic.at(3));

    return intrinsic;
}

t::geometry::PointCloud rgbd_to_pcl(kinect_capture *cap, camera::PinholeCameraIntrinsic intrinsic) {
    t::geometry::Image color = core::Tensor(reinterpret_cast<const uint8_t*>(cap->cv_color_img.data),
                                                {cap->cv_color_img.rows,
                                                cap->cv_color_img.cols, 3},
                                                core::UInt8, gpu_device);

    t::geometry::Image depth = core::Tensor(reinterpret_cast<const uint16_t*>(cap->cv_depth_img.data),
                                                    {cap->cv_depth_img.rows,
                                                    cap->cv_depth_img.cols, 1},
                                                    core::UInt16, gpu_device);

    std::shared_ptr<geometry::RGBDImage> rgbd_image_ptr = geometry::RGBDImage::CreateFromColorAndDepth(color.ToLegacy(), depth.ToLegacy(), 
                                                    depth_scale, 
                                                    depth_trunc,
                                                    convert_rgb_to_intensity);
    geometry::RGBDImage *rgbd_image = rgbd_image_ptr.get();

    
    std::shared_ptr<geometry::PointCloud> cloud = geometry::PointCloud::CreateFromRGBDImage(*rgbd_image, intrinsic);
    t::geometry::PointCloud cloud_t = t::geometry::PointCloud::FromLegacy(*(cloud.get()));

    return cloud_t;
}

geometry::AxisAlignedBoundingBox cloud_range_crop(Eigen::Vector3d min_crop_range, Eigen::Vector3d max_crop_range, Eigen::Vector3d box_color) {
    geometry::AxisAlignedBoundingBox bbox = geometry::AxisAlignedBoundingBox(min_crop_range, max_crop_range);
    bbox.color_ = box_color;

    return bbox;
}