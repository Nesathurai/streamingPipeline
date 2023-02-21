#include "open3d/Open3D.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include <kinect_capture.hpp>
#include <opencv2/opencv.hpp>
#include <cstring>
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
#define PORT 60000
#define NUM_THREADS 3

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
} transmit_args_t;

typedef struct
{
    std::vector<t::geometry::Image> *color_img_list;
    std::vector<t::geometry::Image> *depth_img_list;
    int id;
} generateMesh_args_t;

constexpr float voxel_size = 0.011; // default 0.01
constexpr int block_count = 10000;
constexpr float depth_max = 5.0; // default 5.0
constexpr float trunc_voxel_multiplier = 8.0;

static std::vector<core::Tensor> extrinsic_tf_list;
static std::vector<core::Tensor> intrinsic_list;

uint32_t device_count = 1;
std::mutex img_lock;

bool enableDebugging = 0;
bool enableRender = 0;
// char ipAddress[255] = "sc-4.arena.andrew.cmu.edu";
char ipAddress[255] = "169.254.125.169";

int server_fd, new_socket, valread;
struct sockaddr_in address;

circular_buffer<open3d::geometry::TriangleMesh, 100> meshes;

open3d::geometry::TriangleMesh transmitMesh;

// Declaration of thread condition variable
pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;
// declaring mutex
pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;

int open3d_to_draco(open3d::geometry::TriangleMesh *inOpen3d, draco::EncoderBuffer *outDracoBuffer)
{
    draco::Mesh *open3dToDracoMesh = new draco::Mesh();
    open3dToDracoMesh->set_num_points((uint32_t)inOpen3d->vertices_.size());
    open3dToDracoMesh->SetNumFaces(inOpen3d->triangles_.size());
    draco::PointAttribute open3dToDracoPositionAttribute;
    draco::PointAttribute open3dToDracoNormalAttribute;

    // type based off enum in geometry_attribute.h
    open3dToDracoPositionAttribute.Init(draco::PointAttribute::Type(0), 3, draco::DataType::DT_FLOAT64, true, inOpen3d->vertices_.size());
    open3dToDracoNormalAttribute.Init(draco::PointAttribute::Type(1), 3, draco::DataType::DT_FLOAT64, true, inOpen3d->vertex_normals_.size());

    // add these attributes to the mesh and check if it succeded
    int32_t open3dToDracoPositionAttributeID = open3dToDracoMesh->AddAttribute(open3dToDracoPositionAttribute, true, (uint32_t)inOpen3d->vertices_.size());
    if (open3dToDracoPositionAttributeID == -1)
    {
        printf("open3d to draco position attribute adding failed\n");
        return -1;
    }
    int32_t open3dToDracoNormalAttributeID = open3dToDracoMesh->AddAttribute(open3dToDracoNormalAttribute, true, (uint32_t)inOpen3d->vertex_normals_.size());
    if (open3dToDracoNormalAttributeID == -1)
    {
        printf("open3d to draco normal attribute adding failed\n");
        return -1;
    }

    // initialize attribute vertex values from open3d
    unsigned int inOpen3dAttributeIdx = 0;
    for (auto itr = inOpen3d->vertices_.begin(); itr != inOpen3d->vertices_.end(); itr++)
    {
        if (enableDebugging)
        {
            std::cout << (*itr)[0] << ", " << (*itr)[1] << ", " << (*itr)[2] << "\n";
        }
        double inOpen3dVertex[3];
        inOpen3dVertex[0] = (*itr)[0];
        inOpen3dVertex[1] = (*itr)[1];
        inOpen3dVertex[2] = (*itr)[2];
        open3dToDracoMesh->attribute(open3dToDracoPositionAttributeID)->SetAttributeValue(draco::AttributeValueIndex(inOpen3dAttributeIdx), &inOpen3dVertex[0]);
        inOpen3dAttributeIdx++;
    }

    inOpen3dAttributeIdx = 0;
    for (auto itr = inOpen3d->vertex_normals_.begin(); itr != inOpen3d->vertex_normals_.end(); itr++)
    {
        if (enableDebugging)
        {
            std::cout << (*itr)[0] << ", " << (*itr)[1] << ", " << (*itr)[2] << "\n";
        }
        double inOpen3dVertex[3];
        inOpen3dVertex[0] = (*itr)[0];
        inOpen3dVertex[1] = (*itr)[1];
        inOpen3dVertex[2] = (*itr)[2];
        open3dToDracoMesh->attribute(open3dToDracoNormalAttributeID)->SetAttributeValue(draco::AttributeValueIndex(inOpen3dAttributeIdx), &inOpen3dVertex[0]);
        inOpen3dAttributeIdx++;
    }

    // faces look to be successfully added to the open3dtodracomesh -> failure must be somewhere else TODO
    for (unsigned long i = 0; i < inOpen3d->triangles_.size(); ++i)
    {
        // adding faces broken
        // const draco::Mesh::Face tmpFace({draco::PointIndex((uint32_t)inOpen3d->triangles_[i][0]),draco::PointIndex((uint32_t)inOpen3d->triangles_[i][1]),draco::PointIndex((uint32_t)inOpen3d->triangles_[i][2])});
        draco::Mesh::Face tmpFace = draco::Mesh::Face();
        tmpFace[0] = draco::PointIndex((uint32_t)inOpen3d->triangles_[i][0]);
        tmpFace[1] = draco::PointIndex((uint32_t)inOpen3d->triangles_[i][1]);
        tmpFace[2] = draco::PointIndex((uint32_t)inOpen3d->triangles_[i][2]);

        // face already initialized when face size is set
        open3dToDracoMesh->SetFace(draco::FaceIndex((uint32_t)i), tmpFace);
    }

    // draco::EncoderBuffer meshBuffer;
    // draco::Mesh *meshToSave = nullptr;
    // draco::Mesh *mesh = nullptr;

    // draco::StatusOr<std::unique_ptr<draco::Mesh>> maybe_mesh = draco::ReadMeshFromFile("/home/allan/draco_encode_cpp/custom0.obj", false);
    // if (!maybe_mesh.ok())
    // {
    //     printf("Failed loading the input mesh: %s.\n", maybe_mesh.status().error_msg());
    //     throw std::exception();
    // }

    // mesh = maybe_mesh.value().get();
    // mesh = open3dToDracoMesh;
    // pc = std::move(maybe_mesh).value();
    // pc = maybe_mesh.value().get();

    // Convert compression level to speed (that 0 = slowest, 10 = fastest).
    const int speed = 10 - 1;
    // const int speed = 0;

    draco::Encoder encoder;
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 10);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL, 10);
    encoder.SetSpeedOptions(speed, speed);

    // const bool input_is_mesh = mesh && mesh->num_faces() > 0;

    // Convert to ExpertEncoder that allows us to set per-attribute options.
    // std::unique_ptr<draco::ExpertEncoder> expert_encoder;
    // if (input_is_mesh)
    // {
    // expert_encoder.reset(new draco::ExpertEncoder(*open3dToDracoMesh));
    // }
    // else
    // {
    //     expert_encoder.reset(new draco::ExpertEncoder(*pc));
    // }
    // expert_encoder->Reset(encoder.CreateExpertEncoderOptions(*pc));

    // if (input_is_mesh)
    // {
    // std::cout << "about to encode" << std::endl;

    encoder.EncodeMeshToBuffer(*open3dToDracoMesh, outDracoBuffer);
    if (outDracoBuffer != NULL)
    {
        return 0;
    }
    return -1;
}

void transmitData(open3d::geometry::TriangleMesh *inOpen3d, int counter)
{
    draco::EncoderBuffer meshBuffer;
    char buffer[MAXTRANSMIT] = {0};

    if (open3d_to_draco(inOpen3d, &meshBuffer) == 0)
    {
        if ((meshBuffer.size() > 100))
        {
            printf("(%d) server to send: %ld\n", counter, meshBuffer.size());
            if (sprintf(buffer, "%ld", meshBuffer.size()) <= 0)
            {
                printf("sprintf size to buffer FAILED\n");
            }

            // printf("(%d) server: transfer started; return: %d\n", counter, valsent);
            ssize_t valsent = send(new_socket, buffer, MAXTRANSMIT, 0);
            if (valsent == MAXTRANSMIT)
            {
                int totalSent = 0;
                int toTransfer = meshBuffer.size();
                // connection established, start transmitting
                // TODO: could memory allocation be causing slowdown?
                char outBuffer[meshBuffer.size()] = {0};
                auto result = copy(meshBuffer.buffer()->begin(), meshBuffer.buffer()->end(), outBuffer);
                // check that the copy was successful
                assert(result - outBuffer == meshBuffer.buffer()->end() - meshBuffer.buffer()->begin());

                while (toTransfer > 0)
                {
                    if (toTransfer > MAXTRANSMIT)
                    {
                        valsent = send(new_socket, outBuffer + totalSent, MAXTRANSMIT, 0);
                    }
                    else
                    {
                        valsent = send(new_socket, outBuffer + totalSent, toTransfer, 0);
                    }
                    if (valsent == -1)
                    {
                        printf("valsent = -1 -> socket error 2\n");
                        return;
                    }
                    toTransfer -= valsent;
                    totalSent += valsent;
                }
            }
            else
            {
                printf("valsent = -1 or less than MAXTRANSMIT-> socket error 1\n");
                return;
            }
        }
        else
        {
            printf("frame capture failed (meshBuffer is less than 100 bytes)\n");
            return;
        }
    }
    else
    {
        printf("open3d to draco failed - return -1\n");
        return;
    }
}

static void *transmitDataWrapper(void *data)
{

    args_t *args = (args_t *)data;

    int opt = 1;
    int addrlen = sizeof(address);

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    struct hostent *hp;
    hp = gethostbyname(ipAddress);
    address.sin_family = hp->h_addrtype;
    bcopy((char *)hp->h_addr, (char *)&address.sin_addr, hp->h_length);
    address.sin_port = htons(PORT);

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    int counter = 0;

    // only transmit when signal is recieved
    // while (1)
    while (1)
    {
        if (!meshes.empty())
        {
            // pthread_cond_wait(&cond1, &lock1);
            // with 0 decimation, transmission takes about 0.02s - 50fps
            geometry::TriangleMesh legacyMesh = meshes.get().value();
            if (&legacyMesh != NULL)
            {
                transmitData(&(legacyMesh), counter);
            }
            else
            {
                printf("trying to send data, but mesh is null\n");
            }
            counter++;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    // closing the connected socket
    close(new_socket);
    // closing the listening socket
    shutdown(server_fd, SHUT_RDWR);
    return NULL;
}

void generateMeshAndTransmit(std::vector<t::geometry::Image> *color_img_list, std::vector<t::geometry::Image> *depth_img_list)
{
    visualization::rendering::Material material("defaultLit");
    material.SetDefaultProperties();

    t::geometry::TriangleMesh mesh;
    t::geometry::Image texture_img;

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    high_resolution_clock::time_point fpsStart = high_resolution_clock::now();
    duration<double, std::milli> fpsDel;
    int fpsCounter = 0;
    delta("frame capture period");
    while (1)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(30));
        if (img_lock.try_lock())
        {
            // TODO: some sort of slowdown here 
            // t::geometry::VoxelBlockGrid voxel_grid({"tsdf", "weight"},
            //                                        {core::Dtype::Float32, core::Dtype::Float32},
            //                                        {{1}, {1}}, voxel_size, 16, block_count, gpu_device);
            t::geometry::VoxelBlockGrid voxel_grid({"tsdf", "weight"},
                                                   {core::Dtype::Float32, core::Dtype::Float32},
                                                   {{1}, {1}}, voxel_size, 16, block_count, gpu_device);
            // texture_img = color_img_list->at(0);
            for (int i = 0; i < device_count; i++)
            {
                core::Tensor frustum_block_coords = voxel_grid.GetUniqueBlockCoordinates(depth_img_list->at(0), intrinsic_list.at(0),
                                                                                         extrinsic_tf_list.at(0), depth_scale,
                                                                                         depth_max, trunc_voxel_multiplier);
                voxel_grid.Integrate(frustum_block_coords, depth_img_list->at(0), intrinsic_list.at(0), extrinsic_tf_list.at(0),
                                     depth_scale, depth_max, trunc_voxel_multiplier);
            }
            img_lock.unlock();
            

            mesh = voxel_grid.ExtractTriangleMesh(0, -1).To(cpu_device);
            mesh.RemoveVertexAttr("normals");
            // material.SetAlbedoMap(texture_img);
            // mesh.SetMaterial(material);

            // convert from tensor to legacy (regular triangle mesh)
            // use semaphore because socket thread is waiting
            transmitMesh = mesh.ToLegacy();
            if (transmitMesh.vertices_.size() != 0)
            {
                // std::shared_ptr<open3d::geometry::TriangleMesh> decimated = transmitMesh.SimplifyVertexClustering(0.01);
                // decimated = decimated->SimplifyQuadricDecimation(0.5*decimated->triangles_.size(), 1000000000, 1.0);
                // std::shared_ptr<open3d::geometry::TriangleMesh> decimated = transmitMesh.SimplifyQuadricDecimation(0.5*transmitMesh.triangles_.size(), 1000000000, 1.0);
                // meshes.put(*(decimated.get()));
                meshes.put(transmitMesh);
                // pthread_cond_signal(&cond1);
            }
            else
            {
                printf("conversion to legacy mesh FAILED\n");
            }
            delta("frame time");
            if(fpsCounter % 30 == 0){
                fpsDel = (high_resolution_clock::now()-fpsStart)/1000;
                printf("fps: %f\n",((30)/(fpsDel.count())));
                fpsStart = high_resolution_clock::now();
            }
            fpsCounter++;
        }
        else{
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
    }
}

static void *generateMeshAndTransmitWrapper(void *data)
{
    // calls generate mesh and transmit (the function that grabs frames that were generated and converts to mesh)
    generateMesh_args_t *args = (generateMesh_args_t *)data;
    generateMeshAndTransmit(args->color_img_list, args->depth_img_list);
    return NULL;
}

void startCam(MultiKinectCapture *multi_cap, std::vector<t::geometry::Image> *color_img_list,
              std::vector<t::geometry::Image> *depth_img_list,
              std::vector<cv::Mat> *cv_color_img_list, std::vector<cv::Mat> *cv_depth_img_list)
{
    std::cout << "Found " << k4a_device_get_installed_count() << " connected devices" << std::endl;

    multi_cap = new MultiKinectCapture(device_count);

    camera::PinholeCameraIntrinsic intrinsic;
    for (uint32_t i = 0; i < device_count; i++)
    {
        std::cout << multi_cap->capture_devices.at(i)->serial_num << std::endl;
        intrinsic = get_camera_intrinsic(multi_cap->capture_devices.at(i));
        core::Tensor intrinsic_t = core::eigen_converter::EigenMatrixToTensor(intrinsic.intrinsic_matrix_);
        intrinsic_list.push_back(intrinsic_t);
    }

    multi_cap->get_synchronized_captures();

    // use get_camera_tf() if looking at Apriltag
    extrinsic_tf_list.push_back(core::Tensor::Eye(4, core::Dtype::Float64, cpu_device));
    // extrinsic_tf_list = get_camera_tf(multi_cap);

    while (1)
    {
        if (img_lock.try_lock())
        {
            get_synced_images(multi_cap, color_img_list, depth_img_list,
                              cv_color_img_list, cv_depth_img_list);
            img_lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

static void *startCamWrapper(void *data)
{
    start_cam_args_t *start_cam_args = (start_cam_args_t *)data;
    startCam(start_cam_args->multi_cap, start_cam_args->color_img_list, start_cam_args->depth_img_list, start_cam_args->cv_color_img_list, start_cam_args->cv_depth_img_list);
    return NULL;
}

int main(int argc, char **argv)
{
    pthread_t threads[NUM_THREADS];
    args_t args[NUM_THREADS];

    //////////////////////////
    //     start cameras    //
    //////////////////////////
    MultiKinectCapture *multi_cap;
    std::vector<t::geometry::Image> color_img_list(device_count);
    std::vector<t::geometry::Image> depth_img_list(device_count);
    std::vector<cv::Mat> cv_color_img_list(device_count);
    std::vector<cv::Mat> cv_depth_img_list(device_count);

    start_cam_args_t start_cam_args;
    start_cam_args.multi_cap = multi_cap;
    start_cam_args.color_img_list = &color_img_list;
    start_cam_args.depth_img_list = &depth_img_list;
    start_cam_args.cv_color_img_list = &cv_color_img_list;
    start_cam_args.cv_depth_img_list = &cv_depth_img_list;
    start_cam_args.id = 0;
    pthread_create(&threads[0], NULL, startCamWrapper, &start_cam_args);

    ///////////////////////////
    // create server socket  //
    ///////////////////////////
    transmit_args_t transmit_args;
    transmit_args.port = PORT;
    transmit_args.id = 1;
    pthread_create(&threads[1], NULL, transmitDataWrapper, &transmit_args);

    ///////////////////////////
    // get new frames, mesh  //
    ///////////////////////////
    generateMesh_args_t generateMesh_args;
    generateMesh_args.color_img_list = &color_img_list;
    generateMesh_args.depth_img_list = &depth_img_list;
    generateMesh_args.id = 2;
    pthread_create(&threads[2], NULL, generateMeshAndTransmitWrapper, &generateMesh_args);

    //////////////////////////
    // visualization window //
    //////////////////////////
    // to reset display output: export DISPLAY=":0.0"
    // visualization::gui::Application &o3d_app = visualization::gui::Application::GetInstance();
    // o3d_app.Initialize("/home/sc/Open3D-0.16.0/build/bin/resources");
    // std::shared_ptr<visualization::visualizer::O3DVisualizer> visualizer = std::make_shared<visualization::visualizer::O3DVisualizer>("visualization", 3840, 2160);

    // Eigen::Vector4f bg_color = {1.0, 1.0, 1.0, 1.0};
    // visualizer->SetBackground(bg_color);
    // visualizer->ShowSettings(true);
    // visualizer->ResetCameraToDefault();
    // visualization::gui::Application::GetInstance().AddWindow(visualizer);

    ///////////////////////////
    // reconstruction thread //
    ///////////////////////////

    // o3d_app.Run();

    for (unsigned i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    return 1;
}
