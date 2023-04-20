#include "open3d/Open3D.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include <kinect_capture.hpp>
#include <opencv2/opencv.hpp>
#include <k4a/k4a.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>
#include <camera_alignment.hpp>
#include <Eigen/Core>
#include <multi_kinect_capture.hpp>
#include <util.hpp>
#include <texture_mapping.hpp>
#include <chrono>
#include <thread>

#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <netdb.h>
#include <inttypes.h>

#include "draco/compression/encode.h"
#include "draco/core/cycle_timer.h"
#include "draco/io/file_utils.h"
#include "draco/io/mesh_io.h"
#include "draco/io/point_cloud_io.h"
#include "draco/io/obj_encoder.h"
#include "utils.h"
#include "open3d/Open3D.h"
#include "circularBuffer.hpp"

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
#define MAXTRANSMIT 1500
#define PORT 8080
#define NUM_THREADS 2

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::system_clock;
using std::chrono::time_point;
high_resolution_clock::time_point start_time = high_resolution_clock::now();
int time_first = 1;

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

typedef struct
{
    int port;
    int id;
} args_t;

typedef struct
{
    MultiKinectCapture *multi_cap;
    std::vector<t::geometry::Image> *color_img_list;
    std::vector<t::geometry::Image> *depth_img_list;
    std::vector<cv::Mat> *cv_color_img_list;
    std::vector<cv::Mat> *cv_depth_img_list;
    int id;
} start_cam_args_t;

typedef struct
{
    open3d::geometry::TriangleMesh *inOpen3d;
    int counter;
    int port;
    int id;
    int totalImageFrames;
    int totalDepthFrames;
    int cameraIdx;
} transmit_args_t;

typedef struct
{
    std::vector<t::geometry::Image> *color_img_list;
    std::vector<t::geometry::Image> *depth_img_list;
    MultiKinectCapture *multi_cap;
    int id;
} generateMesh_args_t;

float voxel_size = 0.01; // default 0.01 - optimal is 0.011 in meters
constexpr int block_count = 10000;
constexpr float depth_max = 5.0; // default 5.0
constexpr float trunc_voxel_multiplier = 8.0;
int totalFrames[] = {-1, -1, -1, -1};
int totalImageFrames[] = {300, 300, 300, 300};
int totalDepthFrames[] = {300, 300, 300, 300};

static std::vector<core::Tensor> extrinsic_tf_list;
static std::vector<core::Tensor> intrinsic_list;

uint32_t device_count = 4;
std::mutex img_lock;

bool enableDebugging = 0;
bool enableRender = 0;
// char ipAddress[255] = "sc-4.arena.andrew.cmu.edu";
char ipAddress[255] = "169.254.125.169";

int server_fd, new_socket, valread;
struct sockaddr_in address;

circular_buffer<open3d::geometry::TriangleMesh, 100> meshes;
circular_buffer<cv::Mat, 500> imageBuffer[4];
circular_buffer<cv::Mat, 500> depthBuffer[4];
circular_buffer<k4a::image, 500> depthBufferBin[4];
int saveThreadsFinished = 0;
int depthWidth = 640;
int depthHeight = 576;
int cameras = 4;
int framesPerSetting = 100;

int main(int argc, char **argv)
{
    // set parameters for scripting
    if (argc <= 1)
    {
        std::cerr << "Usage: " << argv[0] << " framesPerSetting " << std::endl;
    }
    else if (argc == 2)
    {
        framesPerSetting = std::stoi(argv[1]);
        std::cout << "Number of frames: " << framesPerSetting << std::endl;
    }

    char frameInStr[1024] = {0};
    char frameOutStr[1024] = {0};
    char allFramesOutStr[1024] = {0};

    size_t frameSizeIn = depthWidth * depthHeight;
    size_t frameSizeOut = frameSizeIn * cameras;
    size_t allFramesSize = frameSizeOut * framesPerSetting;
    size_t allBytesAdded = 0;
    uint16_t *allFramesBuffer = (uint16_t *)malloc(allFramesSize * sizeof(uint16_t));

    std::ofstream allDepthBin("/home/sc/streamingPipeline/analysisData/ref/allDepthBin", std::ios::binary);

    int frame_count = 0;

    for (int i = 0; i < framesPerSetting; i++)
    {
        // each frame output needs to be a 2x2 arrangement of input
        uint16_t *rowBuffer = (uint16_t *)malloc(depthWidth * sizeof(uint16_t));
        uint16_t *frameBufferIn = (uint16_t *)malloc(frameSizeIn * sizeof(uint16_t));
        uint16_t *frameBufferOut = (uint16_t *)malloc(frameSizeOut * sizeof(uint16_t));
        sprintf(frameOutStr, "/home/sc/streamingPipeline/analysisData/ref/frame_%d_depthBin", 0);
        std::ofstream frameOut(frameOutStr, std::ios::binary);

        for (int j = 0; j < cameras; j++)
        {
            // std::cout << j << std::endl;
            // read in individual frame_i_camera_j binary
            sprintf(frameInStr, "/home/sc/streamingPipeline/analysisData/ref/frame_%d_camera_%d_depthBin", i, j);
            std::ifstream frameIn(frameInStr, std::ios::binary);
            frameIn.read(reinterpret_cast<char *>(frameBufferIn), frameSizeIn * sizeof(uint16_t));
            frameIn.close();
            for (int row = 0; row < depthHeight; row++)
            {
                // write to output based on the camera position
                if (j == 0)
                {
                    // std::cout << 2*row*depthWidth << std::endl;
                    memcpy(&frameBufferOut[2 * row * depthWidth], &frameBufferIn[row * depthWidth], depthWidth* sizeof(uint16_t));
                }
                else if (j == 1)
                {
                    // std::cout << depthWidth + 2*row*depthWidth << std::endl;
                    memcpy(&frameBufferOut[depthWidth + 2 * row * depthWidth], &frameBufferIn[row * depthWidth], depthWidth* sizeof(uint16_t));
                }
                else if (j == 2)
                {
                    // std::cout << 2*depthHeight*depthWidth + 2*row*depthWidth << std::endl;
                    memcpy(&frameBufferOut[2 * depthHeight * depthWidth + 2 * row * depthWidth], &frameBufferIn[row * depthWidth], depthWidth* sizeof(uint16_t));
                }
                else if (j == 3)
                {
                    // std::cout << 2*depthHeight*depthWidth + depthWidth + 2*row*depthWidth << std::endl;
                    memcpy(&frameBufferOut[2 * depthHeight * depthWidth + depthWidth + 2 * row * depthWidth], &frameBufferIn[row * depthWidth], depthWidth* sizeof(uint16_t));
                }
            }
            // frameOut.write(reinterpret_cast<char *>(frameBufferOut+j*640*576), frameSizeOut);
        }
        
        // cv::Mat tmp(576, 640, CV_16UC1, frameBufferIn);
        // char outPath[1024 * 2] = {0};
        // sprintf(outPath, "/home/sc/streamingPipeline/analysisData/trvl/%d.png", frame_count);
        // std::cout << outPath << std::endl;
        // cv::imwrite(outPath, tmp);


        cv::Mat tmp(576*2, 640*2, CV_16UC1, frameBufferOut);
        char outPath[1024 * 2] = {0};
        sprintf(outPath, "/home/sc/streamingPipeline/analysisData/trvl/%d.png", frame_count);
        std::cout << outPath << std::endl;
        cv::imwrite(outPath, tmp);

        frameOut.write(reinterpret_cast<char *>(frameBufferOut), frameSizeOut * sizeof(uint16_t));
        frameOut.close();

        allDepthBin.write(reinterpret_cast<char *>(frameBufferOut), frameSizeOut* sizeof(uint16_t));
        std::cout << allBytesAdded << " bytes" << std::endl;
        allBytesAdded += frameSizeOut;
        frame_count++;
        // now that a single frame has been created (from 4 camera depth images), 
        //   they must be appended together to create a video stream
    }
    std::cout << "frame count\n" << frame_count << std::endl;
    allDepthBin.close();
    

    // read in each individual frame
    // depthBufferBin[i].put(multi_cap->capture_devices.at(i)->depthImage);
    // memcpy(&allFramesBinBuffer[bytesAdded], &multi_cap->capture_devices.at(i)->depthImage, depthWidth * depthHeight);
    // bytesAdded += depthWidth * depthHeight;
    // }

    // sprintf(outDepthBinAll, "/home/sc/streamingPipeline/analysisData/ref/frame_%d_depthBin", depthBinAllIdx);
    // k4a::image depthBinAll = allFramesBin;
    // std::ofstream saveDepthBinAll(outDepthBinAll, std::ios::binary);
    // saveDepthBinAll.write(reinterpret_cast<char *>(allFramesBinBuffer), bytesAdded);
    // saveDepthBinAll.close();
    // depthBinAllIdx++;

    return 1;
}
