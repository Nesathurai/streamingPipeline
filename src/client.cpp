// Client side C/C++ program to demonstrate Socket
// programming
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <pthread.h>
#include <unordered_map>
#include <netdb.h>

#include "draco/compression/encode.h"
#include "draco/core/cycle_timer.h"
#include "draco/io/file_utils.h"
#include "draco/io/mesh_io.h"
#include "draco/io/obj_encoder.h"
#include "draco/io/point_cloud_io.h"
#include "draco/mesh/mesh.h"
#include "./utils.h"
#include "open3d/Open3D.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/t/io/TriangleMeshIO.h"
#include "open3d/io/TriangleMeshIO.h"
#include "circularBuffer.hpp"

// #include <connect.h>

using namespace std;
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

// pthread_mutex_t fileMutex;
// int sock = 0;
// int client_fd = 0;
int enableDebugging = 0;
// char ipAddress[255] = "sc-4.arena.andrew.cmu.edu";
char ipAddress0[255] = "169.254.125.169";
char ipAddress1[255] = "169.254.25.1";

// Declaration of thread condition variable
pthread_cond_t cond1 = PTHREAD_COND_INITIALIZER;

// declaring mutex
pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;

circular_buffer<open3d::geometry::TriangleMesh, 100> meshes;
circular_buffer<open3d::geometry::TriangleMesh, 100> meshIngest0;
circular_buffer<open3d::geometry::TriangleMesh, 100> meshIngest1;

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

// TODO: bottleneck is decoding which takes about 0.06 seconds - 14fps
int draco_to_open3d(open3d::geometry::TriangleMesh *outOpen3d, draco::EncoderBuffer *inDracoBuffer)
{
    // delta();
    draco::DecoderBuffer decoderBuffer;
    if (inDracoBuffer->size() <= 0)
    {
        return -1;
    }
    decoderBuffer.Init(inDracoBuffer->data(), inDracoBuffer->size());

    draco::Decoder decoder;
    std::shared_ptr<draco::Mesh> meshToSave = decoder.DecodeMeshFromBuffer(&decoderBuffer).value();
    if (meshToSave.get() != NULL)
    {
        // fixes segfault
        // printf("decoded successfully\n");
    }
    else
    {
        printf("decode FAILED\n");
        return -1;
    }

    // needs at least two attributes: vertices and normals
    if (meshToSave->num_attributes() <= 1)
    {
        printf("in draco to open3d, mesh to save num attributes is less than or equal to 1\n");
        return -1;
    }

    if (enableDebugging)
    {
        printf("mesh num attributes: %d\n", meshToSave->num_attributes());
        cout << "mesh number of faces: " << meshToSave->num_faces() << std::endl;
        cout << "mesh number of points: " << meshToSave->num_points() << std::endl;

        // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
        // Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
        // std::cout << "mesh triangles dimension: " << mesh_ptr->triangles_.begin().base()->NumDimensions << std::endl;

        // vector to hold all vertices info
        std::cout << "draco format: " << std::endl;
    }

    std::vector<Eigen::Vector3d> allVerticesP;
    std::vector<Eigen::Vector3d> allVerticesN;
    const draco::PointAttribute *attr;

    unsigned long count0 = 0;

    for (int i = 0; i < meshToSave->num_attributes(); i++)
    {
        attr = meshToSave->GetAttributeByUniqueId(i);
        if (enableDebugging)
        {
            std::cout << attr->TypeToString(attr->attribute_type()) << std::endl;
            std::cout << "size: " << attr->size() << std::endl;
            std::cout << "size indices map size: " << attr->indices_map_size() << std::endl;
            count0 = 0;
        }

        for (unsigned long j = 0; (j < attr->size()); j++)
        {
            // grab vertex data from draco
            // need to convert value to double before extracting
            // draco loop
            double dracoVertex[3];
            attr->ConvertValue((draco::AttributeValueIndex((unsigned int)j)), 3, &dracoVertex[0]);
            // dump data into open3d
            // open3d can accept vertex, vertex normal, and vertex color
            Eigen::Vector3d newOpen3dVertex(dracoVertex[0], dracoVertex[1], dracoVertex[2]);

            // only considering position - conains all points (not sure what generic is - same number as position)
            if ((strcmp(attr->TypeToString(attr->attribute_type()).c_str(), "POSITION") == 0))
            {
                allVerticesP.push_back(newOpen3dVertex);
                if (enableDebugging)
                {
                    std::cout << "(" << count0 << ") " << newOpen3dVertex.x() << " " << newOpen3dVertex.y() << " " << newOpen3dVertex.z() << std::endl;
                }
            }
            else if ((strcmp(attr->TypeToString(attr->attribute_type()).c_str(), "NORMAL") == 0))
            {
                allVerticesN.push_back(newOpen3dVertex);
                if (enableDebugging)
                {
                    std::cout << "(" << count0 << ")  " << newOpen3dVertex.x() << " " << newOpen3dVertex.y() << " " << newOpen3dVertex.z() << std::endl;
                }
            }
            else
            {
                if (enableDebugging)
                {
                    std::cout << "attribute not POSITION or NORMAL" << std::endl;
                }
            }
            if (enableDebugging)
            {
                count0++;
            }
        }
    }

    // create a new vector of vectors to store real point data
    // fourth Eigen element (vertex) is the face normal
    std::vector<std::vector<Eigen::Vector3d>> facesWithVerticesAndNormal;
    if (enableDebugging)
    {
        std::cout << "FACES: " << meshToSave->num_faces() << std::endl;
    }
    // loop draco faces and convert every index to the actual vertex value

    // usually, this is not always true
    // attribute 0 = position - MESH_CORNER_ATTRIBUTE
    // attribute 1 = tex_coord - MESH_CORNER_ATTRIBUTE - ignore this
    // attribute 2 = normal - MESH_CORNER_ATTRIBUTE - don't think I need this
    // attribute 3 = generic - MESH_CORNER_ATTRIBUTE - has garbage data
    const draco::PointAttribute *attrP = meshToSave->GetAttributeByUniqueId(0);
    // const draco::PointAttribute *attrN = meshToSave->GetAttributeByUniqueId(1);
    for (unsigned long i = 0; (i < meshToSave->num_faces()); i++)
    {
        // grab draco faces
        // find the coordinate values by indexing all the vertices
        draco::Mesh::Face face = meshToSave->face((draco::FaceIndex((uint32_t)i)));

        double vP0[3];
        double vP1[3];
        double vP2[3];

        attrP->ConvertValue(attrP->mapped_index(face[0]), 3, &vP0[0]);
        attrP->ConvertValue(attrP->mapped_index(face[1]), 3, &vP1[0]);
        attrP->ConvertValue(attrP->mapped_index(face[2]), 3, &vP2[0]);
        if (enableDebugging)
        {
            std::cout << "(" << attrP->mapped_index(face[0]).value() << ")"
                      << " ";
            std::cout << vP0[0] << " " << vP0[1] << " " << vP0[2] << " ";

            std::cout << "(" << attrP->mapped_index(face[1]).value() << ")"
                      << " ";
            std::cout << vP1[0] << " " << vP1[1] << " " << vP1[2] << " ";

            std::cout << "(" << attrP->mapped_index(face[2]).value() << ")"
                      << " ";
            std::cout << vP2[0] << " " << vP2[1] << " " << vP2[2] << " ";
        }

        Eigen::Vector3d tmpX(vP0[0], vP0[1], vP0[2]);
        Eigen::Vector3d tmpY(vP1[0], vP1[1], vP1[2]);
        Eigen::Vector3d tmpZ(vP2[0], vP2[1], vP2[2]);

        std::vector<Eigen::Vector3d> facePN;

        // now put these temporary indecies into a vector and store in all faces
        facePN.push_back(tmpX);
        facePN.push_back(tmpY);
        facePN.push_back(tmpZ);

        // now place the new face in all faces
        facesWithVerticesAndNormal.push_back(facePN);
    }

    // now put all the vertices in the right place
    // manual copying of data
    for (auto itr = allVerticesP.begin(); itr != allVerticesP.end(); itr++)
    {
        Eigen::Vector3d tmpVec((*itr)[0], (*itr)[1], (*itr)[2]);
        outOpen3d->vertices_.push_back(tmpVec);
        // std::cout << itr->x() << " " << itr->y() << " " << itr->z() << std::endl;
    }
    for (auto itr = allVerticesN.begin(); itr != allVerticesN.end(); itr++)
    {
        Eigen::Vector3d tmpVec((*itr)[0], (*itr)[1], (*itr)[2]);
        outOpen3d->vertex_normals_.push_back(tmpVec);
        // std::cout << itr->x() << " " << itr->y() << " " << itr->z() << std::endl;
    }
    if (enableDebugging)
    {
        std::cout << "outOpen3d vertices size: " << outOpen3d->vertices_.size() << std::endl;
        std::cout << "outOpen3d vertices normal size: " << outOpen3d->vertex_normals_.size() << std::endl;
    }

    // construct hashmap to search for indices
    unordered_map<Eigen::Vector3d, unsigned long, matrix_hash<Eigen::Vector3d>> faceMapP;
    unordered_map<Eigen::Vector3d, unsigned long, matrix_hash<Eigen::Vector3d>> faceMapN;

    unsigned long count = 0;
    for (auto itr = outOpen3d->vertices_.begin(); itr != outOpen3d->vertices_.end(); itr++)
    {
        // std::cout << count << " : ";
        // std::cout << (*itr)[0] << ", " << (*itr)[1] << ", " << (*itr)[2] << "\n";
        Eigen::Vector3d tmpVec((*itr)[0], (*itr)[1], (*itr)[2]);
        faceMapP[tmpVec] = count;
        count++;
    }
    count = 0;
    for (auto itr = outOpen3d->vertex_normals_.begin(); itr != outOpen3d->vertex_normals_.end(); itr++)
    {
        // std::cout << count << " : ";
        // std::cout << (*itr)[0] << ", " << (*itr)[1] << ", " << (*itr)[2] << "\n";
        Eigen::Vector3d tmpVec((*itr)[0], (*itr)[1], (*itr)[2]);
        faceMapN[tmpVec] = count;
        count++;
    }

    // now go and reconstruct the correct triangle meshes
    // loop through all triangles
    for (auto itr = facesWithVerticesAndNormal.begin(); itr != facesWithVerticesAndNormal.end(); itr++)
    {

        // get point A and find its index in the stored vertices
        auto vertexA = faceMapP.find((*itr)[0]);
        if (vertexA == faceMapP.end())
        {
            std::cout << "could not find vector " << std::endl;
            continue;
        }

        // now repeat for B and C
        auto vertexB = faceMapP.find((*itr)[1]);
        if (vertexB == faceMapP.end())
        {
            std::cout << "could not find vector " << std::endl;
            continue;
        }

        auto vertexC = faceMapP.find((*itr)[2]);
        if (vertexC == faceMapP.end())
        {
            std::cout << "could not find vector " << std::endl;
            continue;
        }

        // check if its valid
        if (((!AreSame((*itr)[0].x(), vertexA->first[0])) || (!AreSame((*itr)[0].y(), vertexA->first[1])) || (!AreSame((*itr)[0].z(), vertexA->first[2]))))
        {
            std::cout << "looking for:\n " << (*itr)[0] << " "
                      << "and got: \n"
                      << vertexA->first << std::endl;
        }
        if (((!AreSame((*itr)[1].x(), vertexB->first[0])) || (!AreSame((*itr)[1].y(), vertexB->first[1])) || (!AreSame((*itr)[1].z(), vertexB->first[2]))))
        {
            std::cout << "looking for:\n " << (*itr)[1] << " "
                      << "and got: \n"
                      << vertexB->first << std::endl;
        }
        if (((!AreSame((*itr)[2].x(), vertexC->first[0])) || (!AreSame((*itr)[2].y(), vertexC->first[1])) || (!AreSame((*itr)[2].z(), vertexC->first[2]))))
        {
            std::cout << "looking for:\n " << (*itr)[2] << " "
                      << "and got: \n"
                      << vertexC->first << std::endl;
        }

        // index values of open3d vertices P
        // changing the ordering of the vertices changing the faces??? TODO
        Eigen::Vector3i triP((int)vertexA->second, (int)vertexB->second, (int)vertexC->second);

        // for each triangle vertex P there exists a triangle vertex N (but store its actual value, not index)
        // now store the face with the correct vertex indecies in the mesh
        outOpen3d->triangles_.push_back(triP);
    }

    auto outOpen3d2 = std::make_shared<open3d::geometry::TriangleMesh>();
    open3d::io::ReadTriangleMeshOptions opt;
    // {vertex index}/{vertex texture coordinate index}/{vertex normal index}
    if (enableDebugging)
    {
        open3d::io::ReadTriangleMeshFromOBJ("/home/allan/draco_encode_cpp/custom0_open3d.obj", *outOpen3d2, opt);
        outOpen3d2->ComputeTriangleNormals(false);
        // outOpen3d2->RemoveDuplicatedTriangles();
        // outOpen3d2->ComputeVertexNormals(false);
        // outOpen3d->RemoveDuplicatedVertices();
        // outOpen3d2->RemoveDuplicatedVertices();
    }

    outOpen3d->ComputeVertexNormals(false);
    // outOpen3d->RemoveDuplicatedVertices();
    // outOpen3d->RemoveDuplicatedTriangles();
    // outOpen3d->ComputeTriangleNormals(false);
    // outOpen3d->RemoveDegenerateTriangles();
    // outOpen3d->RemoveNonManifoldEdges();

    if (enableDebugging)
    {
        std::cout << std::endl
                  << "open3d format: " << std::endl;
        std::cout << "VERTICES: " << std::endl;
        count = 0;
        for (auto itr = outOpen3d->vertices_.begin(); itr != outOpen3d->vertices_.end(); itr++)
        {
            std::cout << "(" << count << ")";
            std::cout << (*itr)[0] << " " << (*itr)[1] << " " << (*itr)[2] << "\n";
            count++;
        }
        std::cout << "VERTEX NORMALS: " << std::endl;
        count = 0;
        for (auto itr = outOpen3d->vertex_normals_.begin(); itr != outOpen3d->vertex_normals_.end(); itr++)
        {
            std::cout << "(" << count << ")";
            std::cout << (*itr)[0] << " " << (*itr)[1] << " " << (*itr)[2] << "\n";
            count++;
        }

        std::cout << "FACES: " << outOpen3d->triangles_.size() << std::endl;
        // int count2 =0;
        for (auto itr = outOpen3d->triangles_.begin(); itr != outOpen3d->triangles_.end(); itr++)
        {
            std::cout << "(" << (*itr)[0] << ") " << outOpen3d->vertices_[(*itr)[0]][0] << " " << outOpen3d->vertices_[(*itr)[0]][1] << " " << outOpen3d->vertices_[(*itr)[0]][2] << " ";
            std::cout << "(" << (*itr)[1] << ") " << outOpen3d->vertices_[(*itr)[1]][0] << " " << outOpen3d->vertices_[(*itr)[1]][1] << " " << outOpen3d->vertices_[(*itr)[1]][2] << " ";
            std::cout << "(" << (*itr)[2] << ") " << outOpen3d->vertices_[(*itr)[2]][0] << " " << outOpen3d->vertices_[(*itr)[2]][1] << " " << outOpen3d->vertices_[(*itr)[2]][2] << std::endl;
        }
        std::cout << "FACE NORMALS" << std::endl;
        for (auto itr = outOpen3d->triangle_normals_.begin(); itr != outOpen3d->triangle_normals_.end(); itr++)
        {
            std::cout << (*itr)[0] << " " << (*itr)[1] << " " << (*itr)[2] << std::endl;
        }

        std::cout << std::endl
                  << "open3d format REFERENCE: " << std::endl;
        std::cout << "VERTICES: " << std::endl;
        count = 0;
        for (auto itr = outOpen3d2->vertices_.begin(); itr != outOpen3d2->vertices_.end(); itr++)
        {
            std::cout << "(" << count << ")";
            std::cout << (*itr)[0] << " " << (*itr)[1] << " " << (*itr)[2] << "\n";
            count++;
        }
        std::cout << "VERTEX NORMALS: " << std::endl;
        count = 0;
        for (auto itr = outOpen3d2->vertex_normals_.begin(); itr != outOpen3d2->vertex_normals_.end(); itr++)
        {
            std::cout << "(" << count << ")";
            std::cout << (*itr)[0] << " " << (*itr)[1] << " " << (*itr)[2] << "\n";
            count++;
        }
        std::cout << "FACES: " << outOpen3d2->triangles_.size() << std::endl;
        for (auto itr = outOpen3d2->triangles_.begin(); itr != outOpen3d2->triangles_.end(); itr++)
        {
            std::cout << "(" << (*itr)[0] << ") " << outOpen3d2->vertices_[(*itr)[0]][0] << " " << outOpen3d2->vertices_[(*itr)[0]][1] << " " << outOpen3d2->vertices_[(*itr)[0]][2] << " ";
            std::cout << "(" << (*itr)[1] << ") " << outOpen3d2->vertices_[(*itr)[1]][0] << " " << outOpen3d2->vertices_[(*itr)[1]][1] << " " << outOpen3d2->vertices_[(*itr)[1]][2] << " ";
            std::cout << "(" << (*itr)[2] << ") " << outOpen3d2->vertices_[(*itr)[2]][0] << " " << outOpen3d2->vertices_[(*itr)[2]][1] << " " << outOpen3d2->vertices_[(*itr)[2]][2] << std::endl;
        }

        std::cout << "FACE NORMALS" << std::endl;

        for (auto itr = outOpen3d2->triangle_normals_.begin(); itr != outOpen3d2->triangle_normals_.end(); itr++)
        {
            std::cout << (*itr)[0] << " " << (*itr)[1] << " " << (*itr)[2] << std::endl;
        }

        std::cout << "draco\t->\topen3d\ttriangles\tvertices; " << std::endl;
        std::cout << "x\t\t\t" << meshToSave->num_faces() << "\t\t" << meshToSave->num_points() << std::endl;
        std::cout << "\t\tx\t" << outOpen3d->triangles_.size() << "\t\t" << outOpen3d->vertices_.size() << std::endl;

        std::cout << "draco version: has triangles: " << outOpen3d->HasTriangles() << " has triangle normals: " << outOpen3d->HasTriangleNormals();
        std::cout << " has vertices: " << outOpen3d->HasVertices() << " has vertex normals: " << outOpen3d->HasVertexNormals();
        std::cout << " has adj list: " << outOpen3d->HasAdjacencyList() << " has triangle uvs: " << outOpen3d->HasTriangleUvs() << std::endl;

        std::cout << "open3d version: has triangles: " << outOpen3d2->HasTriangles() << " has triangle normals: " << outOpen3d2->HasTriangleNormals();
        std::cout << " has vertices: " << outOpen3d2->HasVertices() << " has vertex normals: " << outOpen3d2->HasVertexNormals();
        std::cout << " has adj list: " << outOpen3d2->HasAdjacencyList() << " has triangle uvs: " << outOpen3d2->HasTriangleUvs() << std::endl;
        // open3d::io::WriteTriangleMeshToOBJ("/home/allan/draco_encode_cpp/custom0_draco_to_open3d.obj", *outOpen3d, false, false, true, false, false, false);
        // open3d::io::WriteTriangleMeshToOBJ("/home/allan/draco_encode_cpp/custom0_open3d.obj", *outOpen3d2, false, false, true, false, false, false);

        printf("success\n");
    }
    // delta("decoding time");
    // delta("decoding time");
    return 0;
}

static void *recieve(void *data)
{
    args_t *args = (args_t *)data;
    int sock;
    ;
    int client_fd = args->id;
    // int sock = 0;
    // int client_fd =0;
    struct sockaddr_in address;

    // convert hostname to ip address
    struct hostent *hp;
    if (args->id == 0)
    {
        hp = gethostbyname(ipAddress0);
        std::cout << ipAddress0 << std::endl;
        std::cout << args->port << std::endl;
    }
    else if (args->id == 1)
    {
        hp = gethostbyname(ipAddress1);
        std::cout << ipAddress1 << std::endl;
        std::cout << args->port << std::endl;
    }
    else
    {
        printf("bad thread id - no ip address\n");
    }

    std::cout << hp->h_addr << std::endl;
    address.sin_family = hp->h_addrtype;
    bcopy((char *)hp->h_addr, (char *)&address.sin_addr, hp->h_length);
    // address.sin_port = htons(args->port);
    address.sin_port = htons(8080);

    if ((sock = socket(PF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        // return -1;
    }

    if ((client_fd = connect(sock, (struct sockaddr *)&address, sizeof(address))) < 0)
    // if ((client_fd = explain_connect (sock, (struct sockaddr *)&address, sizeof(address))) < 0)
    {
        printf("\nConnection Failed in thread: %d with error: %d\n", args->id, client_fd);
        return NULL;
    }
    // will probably break when the file is too small...
    int counter = 0;
    char buffer[MAXTRANSMIT] = {0};
    high_resolution_clock::time_point fpsStart = high_resolution_clock::now();
    duration<double, std::milli> fpsDel;
    delta();
    while (1)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(30));
        if (counter % 30 == 0)
        {
            fpsDel = (high_resolution_clock::now() - fpsStart) / 1000;
            printf("fps: %f\n", ((30) / (fpsDel.count())));
            fpsStart = high_resolution_clock::now();
        }
        ssize_t valread = read(sock, buffer, MAXTRANSMIT);
        // std::cout << buffer << std::endl;
        if (valread == MAXTRANSMIT)
        {
            // strol more robust than atoi
            // int totalSize;
            long toRead = strtol(buffer, NULL, 10);
            // totalSize = toRead;
            if (toRead < 0)
            {
                printf("strtol failed\n");
                continue;
            }

            if ((toRead <= 100))
            {
                printf("(%d) client read %ld, toRead: %ld\n", counter, valread, toRead);
                printf("recieved package less than 100 bytes - this should never print because its prevented server side\n");
                continue;
            }
            printf("(%d:%d) client toRead: %ld\n", args->id, counter, toRead);
            // printf("(%d) read: %d\n", counter, valread);

            char inBuffer[toRead] = {0};
            ssize_t totalRead = 0;
            while (toRead > 0)
            {
                if (toRead > MAXTRANSMIT)
                {
                    valread = read(sock, (inBuffer + totalRead), MAXTRANSMIT);
                }
                else
                {
                    valread = read(sock, (inBuffer + totalRead), toRead);
                }

                if (valread == -1)
                {
                    printf("valread = -1 -> socket error\n");
                    break;
                }
                totalRead += valread;
                toRead -= valread;
                // printf("(%d) read: %d toRead: %d\n", counter, valread, toRead);
            }

            if (toRead == 0)
            {
                draco::EncoderBuffer inDracoBuffer;
                inDracoBuffer.buffer()->insert(inDracoBuffer.buffer()->end(), &inBuffer[0], &inBuffer[totalRead]);

                // convert draco to open3d
                std::shared_ptr<open3d::geometry::TriangleMesh> outOpen3d = std::make_shared<open3d::geometry::TriangleMesh>();
                bool success = draco_to_open3d(outOpen3d.get(), &inDracoBuffer);
                if (success == 0)
                {
                    if (args->id == 0)
                    {
                        meshIngest0.put(*(outOpen3d.get()));
                    }
                    else if (args->id == 1)
                    {
                        meshIngest1.put(*(outOpen3d.get()));
                    }

                    // meshes.put(*(outOpen3d.get()));
                    // signal that a frame has been captured to renderer
                    pthread_cond_signal(&cond1);
                    if (counter < 10)
                    {
                        char outPath[1024] = {0};
                        sprintf(outPath, "/home/sc/streamingPipeline/meshes/frame_%d.obj", counter);
                        open3d::io::WriteTriangleMeshToOBJ(outPath, *outOpen3d, false, false, true, false, false, false);
                        // printf("(%d) buffer save success: %d\n", counter, !success);
                    }
                    counter++;
                }
                else
                {
                    printf("failed decoding\n");
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            else
            {
                printf("valread = -1 -> socket error 2\n");
            }
        }
        else
        {
            long toRead = strtol(buffer, NULL, 10);
            printf("valread = %ld -> toRead = %ld socket error 3\n", valread, toRead);
            break;
            // continue;
        }
        // delta("frame arrival period");
        // pthread_mutex_unlock(&fileMutex);
    }

    // closing the connected socket
    printf("Last error was: %s\n", get_error_text());
    close(client_fd);
    return NULL;
}

static void *app(void *data)
{
    open3d::visualization::Visualizer visualizer;
    if (!visualizer.CreateVisualizerWindow("Mesh", 1600, 900, 50, 50))
    {
        open3d::utility::LogWarning(
            "[DrawGeometries] Failed creating OpenGL "
            "window.");
        return NULL;
    }
    visualizer.GetRenderOption().point_show_normal_ = false;
    visualizer.GetRenderOption().mesh_show_wireframe_ = false;
    visualizer.GetRenderOption().mesh_show_back_face_ = false;
    visualizer.BuildUtilities();
    visualizer.UpdateWindowTitle();
    open3d::visualization::ViewControl &view_control = visualizer.GetViewControl();

    while (1)
    {
        // wait for a new frame to arrive
        pthread_mutex_lock(&lock1);
        pthread_cond_wait(&cond1, &lock1);
        // printf("rendering\n");
        visualizer.ClearGeometries();
        if (meshes.empty() == false)
        {
            open3d::geometry::TriangleMesh legacyMesh = meshes.get().value();
            // visualizer.AddGeometry({legacyMesh});
            visualizer.AddGeometry({std::make_shared<open3d::geometry::TriangleMesh>(legacyMesh)});
            // visualizer.WaitEvents();
            visualizer.PollEvents();
        }
        else
        {
            printf("trying to display data, but mesh is null\n");
        }
        pthread_mutex_unlock(&lock1);
    }
    visualizer.DestroyVisualizerWindow();
    return NULL;
}

static void *merge(void *data)
{
    int counter = 0;
    // want to merge the meshes ingested by camera 0 and camera 1
    while (1)
    {
        if (!meshIngest0.empty() && !meshIngest1.empty())
        {
            // merge the meshes
            open3d::geometry::TriangleMesh meshMerged = meshIngest0.get().value();
            meshMerged += meshIngest1.get().value();
            meshes.put(meshMerged);
            if (counter < 10)
            {
                char outPath[1024] = {0};
                sprintf(outPath, "/home/sc/streamingPipeline/meshesMerged/frame_01_%d.obj", counter);
                open3d::io::WriteTriangleMeshToOBJ(outPath, meshMerged, false, false, true, false, false, false);
            }
            counter++;
        }
        else
        {
            // wait a bit before trying again
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    return NULL;
}
int main(int argc, char const *argv[])
{

    pthread_t threads[NUM_THREADS + 2];
    args_t args[NUM_THREADS + 2];

    for (int i = 0; i < NUM_THREADS; i++)
    {
        args[i].port = PORT + i;
        args[i].id = i;
        pthread_create(&threads[i], NULL, recieve, &args[i]);
    }
    // sleep(20);
    // create visualization thread
    pthread_create(&threads[NUM_THREADS], NULL, app, NULL);
    pthread_create(&threads[NUM_THREADS+1], NULL, merge, NULL);


    for (unsigned i = 0; i < NUM_THREADS + 2; i++)
    {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
