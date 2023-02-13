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
#define MAXTRANSMIT 4096
#define PORT 8080

constexpr float voxel_size = 0.01;
constexpr int block_count = 10000;
constexpr float depth_max = 5.0;
constexpr float trunc_voxel_multiplier = 8.0;

static std::vector<core::Tensor> extrinsic_tf_list;
static std::vector<core::Tensor> intrinsic_list;

uint32_t device_count = 1;
std::mutex img_lock;
pthread_mutex_t fileMutex;

bool enableDebugging = 0;
char ipAddress[255] = "sc-4.arena.andrew.cmu.edu";

int server_fd, new_socket, valread;
struct sockaddr_in address;

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

    // add these attributes to the mesh
    int32_t open3dToDracoPositionAttributeID = open3dToDracoMesh->AddAttribute(open3dToDracoPositionAttribute, true, (uint32_t)inOpen3d->vertices_.size());
    int32_t open3dToDracoNormalAttributeID = open3dToDracoMesh->AddAttribute(open3dToDracoNormalAttribute, true, (uint32_t)inOpen3d->vertex_normals_.size());

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
    // const int speed = 10 - 1;
    const int speed = 0;

    draco::Encoder encoder;
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 32);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL, 32);
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

    return 0;
}

void transmitData(open3d::geometry::TriangleMesh *inOpen3d, int counter)
{
    if (!pthread_mutex_trylock(&fileMutex))
    {
        draco::EncoderBuffer meshBuffer;
        char buffer[1024] = {0};

        open3d_to_draco(inOpen3d, &meshBuffer);

        if ((meshBuffer.size() > 12))
        {
            printf("(%d) mesh buffer size: %ld\n", counter, meshBuffer.size());
            sprintf(buffer, "%ld", meshBuffer.size());
            int response;
            // printf("(%d) server: transfer started; return: %d\n", counter, response);
            response = send(new_socket, buffer, 1024, 0);
            // printf("(%d) send: %d\n", counter, response);

            // connection established, start transmitting
            char outBuffer[meshBuffer.size()] = {0};
            copy(meshBuffer.buffer()->begin(), meshBuffer.buffer()->end(), outBuffer);

            int seek = 0;
            int toTransfer = meshBuffer.size();

            while (toTransfer >= MAXTRANSMIT)
            {
                response = send(new_socket, outBuffer + seek, MAXTRANSMIT, 0);
                toTransfer -= MAXTRANSMIT;
                seek += MAXTRANSMIT;
                // printf("(%d) send: %d\n", counter, response);
                // printf("(%d) Last error was: %s\n", counter, get_error_text());
            }
            response = send(new_socket, outBuffer + seek, toTransfer, 0);
            printf("(%d) send: %d\n", counter, response);
            printf("Last error was: %s\n", get_error_text());
        }
        else
        {
            // size of 12 for some reason is an invalid frame
            std::cout << "frame capture failed" << std::endl;
        }
        pthread_mutex_unlock(&fileMutex);
    }
}

void UpdateSingleMesh(std::shared_ptr<visualization::visualizer::O3DVisualizer> visualizer, std::vector<t::geometry::Image> *color_img_list,
                      std::vector<t::geometry::Image> *depth_img_list)
{
    visualization::rendering::Material material("defaultLit");
    material.SetDefaultProperties();

    t::geometry::TriangleMesh mesh;
    t::geometry::Image texture_img;

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    int counter = 0;
    while (1)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        if (img_lock.try_lock())
        {
            t::geometry::VoxelBlockGrid voxel_grid({"tsdf", "weight"},
                                                   {core::Dtype::Float32, core::Dtype::Float32},
                                                   {{1}, {1}}, voxel_size, 16, block_count, gpu_device);
            texture_img = color_img_list->at(0);
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

            mesh_uv_mapping(&(mesh), intrinsic_list.at(0), extrinsic_tf_list.at(0));

            material.SetAlbedoMap(texture_img);
            mesh.SetMaterial(material);

            visualization::gui::Application::GetInstance().PostToMainThread(
                visualizer.get(), [visualizer, mesh]()
                {
                            visualizer->RemoveGeometry("mesh");
                            visualizer->AddGeometry("mesh", 
                                                    std::make_shared<t::geometry::TriangleMesh>(mesh));
                            
                            visualizer->PostRedraw(); });
        }

        // decimate the mesh to save space
        // inOpen3d->triangles_.size()
        // mesh = mesh.SimplifyQuadricDecimation(0.8, false);
        // std::cout << "passes decimation" << std::endl;


        // convert from tensor to legacy (regular triangle mesh)
        geometry::TriangleMesh legacyMesh;
        legacyMesh = mesh.ToLegacy();

        // transmit data
        transmitData(&legacyMesh, counter);
        
        // char outPath[1024] = {0};
        // sprintf(outPath, "/home/sc/ws/reconstruction/meshes/capture%d.obj",counter);

        // open3d::t::io::WriteTriangleMesh("/home/sc/ws/reconstruction/meshes/capture0.obj", legacyMesh, false, false, true, false, false, false);
        // open3d::io::WriteTriangleMesh(outPath, legacyMesh, false, false, true, false, false, false);
        // std::cout << "wrote to file (for testing purposes): " << counter << std::endl;
        counter++;
    }
}

void UpdateMultiMesh(std::shared_ptr<visualization::visualizer::O3DVisualizer> visualizer, std::vector<t::geometry::Image> *color_img_list,
                     std::vector<t::geometry::Image> *depth_img_list,
                     std::vector<cv::Mat> *cv_color_img_list, std::vector<cv::Mat> *cv_depth_img_list)
{
    visualization::rendering::Material material("defaultLit");
    material.SetDefaultProperties();
    t::geometry::TriangleMesh mesh;

    std::this_thread::sleep_for(std::chrono::milliseconds(4000));

    while (1)
    {
        if (img_lock.try_lock())
        {
            t::geometry::VoxelBlockGrid voxel_grid({"tsdf", "weight"},
                                                   {core::Dtype::Float32, core::Dtype::Float32},
                                                   {{1}, {1}}, voxel_size, 16, block_count, gpu_device);

            for (int i = 0; i < device_count; i++)
            {
                core::Tensor frustum_block_coords = voxel_grid.GetUniqueBlockCoordinates(depth_img_list->at(i), intrinsic_list.at(i),
                                                                                         extrinsic_tf_list.at(i), depth_scale,
                                                                                         depth_max, trunc_voxel_multiplier);

                voxel_grid.Integrate(frustum_block_coords, depth_img_list->at(i), intrinsic_list.at(i), extrinsic_tf_list.at(i),
                                     depth_scale, depth_max, trunc_voxel_multiplier);
            }

            mesh = voxel_grid.ExtractTriangleMesh(0, -1).To(cpu_device);

            mesh.RemoveVertexAttr("normals");

            optimized_multi_cam_uv(&mesh, intrinsic_list, extrinsic_tf_list, cv_depth_img_list);

            cv::Mat stitched_image;
            cv::hconcat(*cv_color_img_list, stitched_image);

            img_lock.unlock();

            auto pblob = std::make_shared<core::Blob>(
                core::Device(), stitched_image.data, [](void *) {});
            // Create tensor
            core::Tensor data_o3d(
                /*shape=*/{stitched_image.rows, stitched_image.cols, stitched_image.channels()},
                /*stride in elements (not bytes)*/
                {int64_t(stitched_image.step[0] / stitched_image.elemSize1()),
                 int64_t(stitched_image.step[1] / stitched_image.elemSize1()), 1},
                stitched_image.data, core::Dtype::UInt8, pblob);
            t::geometry::Image texture_img(data_o3d);

            material.SetAlbedoMap(texture_img);
            mesh.SetMaterial(material);

            visualization::gui::Application::GetInstance().PostToMainThread(
                visualizer.get(), [visualizer, mesh]()
                {
                            visualizer->RemoveGeometry("mesh");
                            visualizer->AddGeometry("mesh", std::make_shared<t::geometry::TriangleMesh>(mesh));

                            visualizer->PostRedraw(); });
        }
    }
}

void start_cam(MultiKinectCapture *multi_cap, std::vector<t::geometry::Image> *color_img_list,
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

int main(int argc, char **argv)
{
    std::vector<t::geometry::Image> color_img_list(device_count);
    std::vector<t::geometry::Image> depth_img_list(device_count);
    std::vector<cv::Mat> cv_color_img_list(device_count);
    std::vector<cv::Mat> cv_depth_img_list(device_count);

    //////////////////////////
    //     start cameras    //
    //////////////////////////
    MultiKinectCapture *multi_cap;
    std::thread cam_thread(start_cam, multi_cap, &color_img_list, &depth_img_list,
                           &cv_color_img_list, &cv_depth_img_list);

    //////////////////////////
    // visualization window //
    //////////////////////////
    // to reset display output: export DISPLAY=":0.0"
    auto &o3d_app = visualization::gui::Application::GetInstance();
    o3d_app.Initialize("/home/sc/Open3D-0.16.0/build/bin/resources");
    auto visualizer = std::make_shared<visualization::visualizer::O3DVisualizer>("visualization", 3840, 2160);
    Eigen::Vector4f bg_color = {1.0, 1.0, 1.0, 1.0};
    visualizer->SetBackground(bg_color);
    visualizer->ShowSettings(true);
    visualizer->ResetCameraToDefault();

    // visualization::gui::Application::GetInstance().Initialize();
    // visualization::gui::Application::GetInstance().Run();
    visualization::gui::Application::GetInstance().AddWindow(visualizer);

    ///////////////////////////
    // create server socket  //
    ///////////////////////////
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

    ///////////////////////////
    // reconstruction thread //
    ///////////////////////////
    std::thread update_thread(UpdateSingleMesh, visualizer, &color_img_list, &depth_img_list);
    // std::thread update_thread(UpdateMultiMesh, visualizer, &color_img_list, &depth_img_list, &cv_color_img_list, &cv_depth_img_list);

    o3d_app.Run();
    update_thread.join();

    // closing the connected socket
    close(new_socket);
    // closing the listening socket
    shutdown(server_fd, SHUT_RDWR);
    printf("Last error was: %s\n", get_error_text());
    return 1;
}
