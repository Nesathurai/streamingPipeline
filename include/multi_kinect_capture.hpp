#ifndef __MULTI_KINECT_CAPTURE_HPP__
#define __MULTI_KINECT_CAPTURE_HPP__

#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <kinect_capture.hpp>
#include <cstdlib>

#define COLOR_EXPOSURE_USEC 8000
#define POWERLINE_FREQ 2

constexpr std::chrono::microseconds MAX_ALLOWABLE_TIME_OFFSET_ERROR_FOR_IMAGE_TIMESTAMP(500);
constexpr int64_t WAIT_FOR_SYNCHRONIZED_CAPTURE_TIMEOUT = 60000;
constexpr uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;

// static const std::vector<int32_t> sync_timing = {-240, -80, 80, 240};
static const std::vector<int32_t> sync_timing = {-80, 80}; // must be 160 us apart

class MultiKinectCapture
{
public:
    // index 1 is master device
    // rest are subordinate devices
    std::vector<kinect_capture *> capture_devices;

    MultiKinectCapture(uint32_t num_devices, int32_t sync_timing, bool is_master)
    {
        this->num_devices = num_devices;

        if (num_devices == 0)
        {
            std::cerr << "Capturer must be passed at least one camera!" << std::endl;
            exit(1);
        }

        uint32_t master_index = -1;
        if (this->num_devices == 1)
        {
            capture_devices.push_back(new kinect_capture(0));
            kinect_capture *cap = capture_devices.at(0);
            cap->get_intrinsic_calibration();

            // If you want to synchronize cameras, you need to manually set both their exposures
            cap->device.set_color_control(K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                                          K4A_COLOR_CONTROL_MODE_MANUAL,
                                          COLOR_EXPOSURE_USEC);
            // This setting compensates for the flicker of lights due to the frequency of AC power in your region. If
            // you are in an area with 50 Hz power, this may need to be updated (check the docs for
            // k4a_color_control_command_t)
            cap->device.set_color_control(K4A_COLOR_CONTROL_POWERLINE_FREQUENCY,
                                          K4A_COLOR_CONTROL_MODE_MANUAL,
                                          POWERLINE_FREQ);

            cap->device.set_color_control(K4A_COLOR_CONTROL_WHITEBALANCE,
                                          K4A_COLOR_CONTROL_MODE_AUTO,
                                          0);

            if (is_master)
            {
                cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
            } else {
                cap->config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
            }
            
            cap->config.subordinate_delay_off_master_usec = 0;
            // }

            kinect_capture *device = capture_devices.at(0);

            // adjust sync timing
            device->config.depth_delay_off_color_usec = sync_timing;
        }

        start_cameras();
        std::cout << "Cameras initialized" << std::endl;
    }

    ~MultiKinectCapture()
    {
        close_cameras();
        for (uint32_t dev_idx = 0; dev_idx < this->num_devices; dev_idx++)
        {
            delete capture_devices.at(dev_idx);
        }
    }

    uint32_t get_num_devices();

    void start_cameras();

    void get_synchronized_captures();

    void close_cameras();

private:
    uint32_t num_devices;
};

#endif //__MULTI_KINECT_CAPTURE_HPP__
