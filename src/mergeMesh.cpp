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
    std::string mesh0;
    std::string mesh1;
    std::string mesh2;
    std::string mesh3;
    std::string nameStr; 
    std::string vx; 
    // set parameters for scripting
    
    if (argc == 5)
    {
        mesh0 = argv[1];
        mesh1 = argv[2];
        mesh2 = argv[3];
        mesh3 = argv[4];

        // TODO: extract vx size from string name
        // string processing to get the name of the file minus .png suffix
        nameStr = mesh0.substr(mesh0.rfind("frame"), mesh0.size());
        nameStr = nameStr.substr(0, 7);
        vx = mesh0.substr(mesh0.rfind('_') + 1, mesh0.size());
        vx = vx.substr(0, 7);
        // std::cout << nameStr << std::endl;
        // std::cout << vx << std::endl;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " mesh0 mesh1 mesh2 mesh3" << std::endl;
        return -1;
    }

    geometry::TriangleMesh out0;
    geometry::TriangleMesh out1;
    geometry::TriangleMesh out2;
    geometry::TriangleMesh out3;

    open3d::io::ReadTriangleMesh(mesh0, out0);
    open3d::io::ReadTriangleMesh(mesh1, out1);
    open3d::io::ReadTriangleMesh(mesh2, out2);
    open3d::io::ReadTriangleMesh(mesh3, out3);

    out0 += out1;
    out0 += out2;
    out0 += out3;

    char outMesh[1024] = {0};
    sprintf(outMesh, "/home/sc/streamingPipeline/analysisData/%s_vx_%s.obj", nameStr.c_str(), vx.c_str());
    std::cout << outMesh << std::endl;
    open3d::io::WriteTriangleMesh(outMesh, out0);

    return 1;
}
