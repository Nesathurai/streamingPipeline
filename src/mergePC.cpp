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
int main(int argc, char **argv)
{
    std::string PC0;
    std::string PC1;
    std::string PC2;
    std::string PC3;
    std::string nameStr; 
    // set parameters for scripting
    
    if (argc == 5)
    {
        PC0 = argv[1];
        PC1 = argv[2];
        PC2 = argv[3];
        PC3 = argv[4];

        // string processing to get the name of the file minus .png suffix
        nameStr = PC0.substr(PC0.rfind("frame"), PC0.size());
        nameStr = nameStr.substr(0, 7);
        std::cout << nameStr << std::endl;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " PC0 PC1 PC2 PC3" << std::endl;
        return -1;
    }

    t::geometry::PointCloud out0;
    t::geometry::PointCloud out1;
    t::geometry::PointCloud out2;
    t::geometry::PointCloud out3;

    open3d::t::io::ReadPointCloud(PC0, out0);
    open3d::t::io::ReadPointCloud(PC1, out1);
    open3d::t::io::ReadPointCloud(PC2, out2);
    open3d::t::io::ReadPointCloud(PC3, out3);

    out0 = out0.Append(out1);
    out0 = out0.Append(out2);
    out0 = out0.Append(out3);

    // TODO: remove similar points?
    out0.RemoveDuplicatedPoints();
    char outPC[1024] = {0};
    sprintf(outPC, "/home/sc/streamingPipeline/analysisData/meshPCs/%s.ply", nameStr.c_str());
    open3d::t::io::WritePointCloud(outPC, out0);

    return 1;
}
