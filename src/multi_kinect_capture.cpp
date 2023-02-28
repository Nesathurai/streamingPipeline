#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <kinect_capture.hpp>
#include <multi_kinect_capture.hpp>

uint32_t MultiKinectCapture::get_num_devices() {
    return this->num_devices;
}

void MultiKinectCapture::start_cameras() {
    // // start subordinate devices first
    // for (uint32_t i = 1; i < get_num_devices(); i++) {
    //     kinect_capture *cap_subordinate = this->capture_devices.at(i);
    //     cap_subordinate->device.start_cameras(&(cap_subordinate->config));
    // }

    // // start master device last
    // kinect_capture *cap_master = this->capture_devices.at(0);
    // cap_master->device.start_cameras(&(cap_master->config));
    
    // std::cout << "All kinect devices have started." << std::endl;

    kinect_capture *cap = this->capture_devices.at(0);
    cap->device.start_cameras(&(cap->config));
}

void MultiKinectCapture::get_synchronized_captures() {
    kinect_capture *cap_master = this->capture_devices.at(0);
    cap_master->capture_frame();

    uint32_t num_devices = get_num_devices();

    if (num_devices > 1) {
        for (uint32_t i = 1; i < num_devices; i++) {
            kinect_capture *subordinate_device = this->capture_devices.at(i);
            subordinate_device->capture_frame();
        }

        // determine sync bound
        bool have_sync_images = false;
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!have_sync_images) {
            int64_t duration_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
            if (duration_ms > WAIT_FOR_SYNCHRONIZED_CAPTURE_TIMEOUT) {
                std::cerr << "ERROR: Timedout waiting for synchronized captures" << std::endl;
                exit(1);
            }

            // cap_master->capture_frame();
            std::chrono::microseconds master_color_image_time = cap_master->colorImage.get_device_timestamp();

            for (uint32_t i = 1; i < num_devices; i++) {
                kinect_capture *subordinate_device = this->capture_devices.at(i);
                // subordinate_device->capture_frame();
                // if (cap_master->colorImage.is_valid() && subordinate_device->colorImage.is_valid()) {
                    std::chrono::microseconds sub_color_image_time = subordinate_device->colorImage.get_device_timestamp();

                    std::chrono::microseconds expected_sub_image_time =
                        master_color_image_time +
                        std::chrono::microseconds{ subordinate_device->config.subordinate_delay_off_master_usec } + 
                        std::chrono::microseconds{ subordinate_device->config.depth_delay_off_color_usec };
                    std::chrono::microseconds sub_image_time_error = std::chrono::duration_cast<std::chrono::microseconds>(sub_color_image_time - expected_sub_image_time);

                    // std::cout << "i: " << std::to_string(i) << std::endl;
                    // std::cout << "expected_sub_image_time: " << std::to_string(expected_sub_image_time.count()) << " usec" << std::endl;
                    // std::cout << "master_color_image_time: " << std::to_string(master_color_image_time.count()) << " usec" << std::endl;
                    // std::cout << "sub_color_image_time: " << std::to_string(sub_color_image_time.count()) << " usec" << std::endl;
                    // std::cout << "sub_image_time_error: " << std::to_string(sub_image_time_error.count()) << " usec" << std::endl;
                    if (sub_image_time_error < -MAX_ALLOWABLE_TIME_OFFSET_ERROR_FOR_IMAGE_TIMESTAMP) {
                        // for (int j = 1; j < num_devices; j++) {
                        //     this->capture_devices.at(j)->capture_frame();
                        // }
                        subordinate_device->capture_frame();
                        
                        // std::cout << "recapture subordinate device frame" << std::endl;
                        break;
                    } else if (sub_image_time_error > MAX_ALLOWABLE_TIME_OFFSET_ERROR_FOR_IMAGE_TIMESTAMP) {
                        cap_master->capture_frame();
                        // std::cout << "recapture master device frame" << std::endl;
                        break;
                    } else {
                        if (i == this->capture_devices.size() - 1) {
                            have_sync_images = true;
                            // std::cout << "have synced image" << std::endl;
                        }
                    }
                // } else if (!cap_master->colorImage.is_valid()) {
                //     cap_master->capture_frame();
                // } else if (!subordinate_device->colorImage.is_valid()) {
                //     subordinate_device->capture_frame();
                // }
                
            }
        }
    }
}

void MultiKinectCapture::close_cameras() {
    std::cout << "closing " << std::to_string(get_num_devices()) << " devices" << std::endl;
    for (uint32_t i = 0; i < get_num_devices(); i++) {
        capture_devices.at(i)->device.close();
    }
    
    std::cout << "All kinect device have closed." << std::endl;
}