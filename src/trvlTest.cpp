#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <deque>
#include "trvl.h"

short CHANGE_THRESHOLD = 10;
int INVALIDATION_THRESHOLD = 2;
int TRVL = 1;
int APPEND = 0;
// int width = 640 * 2;
// int height = 576 * 2;
int width = 640;
int height = 576;
char *inFile;
char *rootPath;

// https://www.geeksforgeeks.org/operator-overloading-cpp-print-contents-vector-map-pair/
// C++ template to print vector container elements
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::deque<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

class InputFile
{
public:
    InputFile(std::string filename, std::ifstream &&input_stream, int width, int height)
        : filename_(filename), input_stream_(std::move(input_stream)), width_(width), height_(height) {}

    std::string filename() { return filename_; }
    std::ifstream &input_stream() { return input_stream_; }
    int width() { return width_; }
    int height() { return height_; }

private:
    std::string filename_;
    std::ifstream input_stream_;
    int width_;
    int height_;
};

class Timer
{
public:
    Timer() : time_point_(std::chrono::steady_clock::now()) {}
    float milliseconds()
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - time_point_).count() / 1000.0f;
    }

private:
    std::chrono::steady_clock::time_point time_point_;
};

class Result
{
public:
    Result(float average_compression_time, float average_decompression_time, float compression_ratio, float average_psnr)
        : average_compression_time_(average_compression_time), average_decompression_time_(average_decompression_time),
          compression_ratio_(compression_ratio), average_psnr_(average_psnr) {}

    float average_compression_time() { return average_compression_time_; }
    float average_decompression_time() { return average_decompression_time_; }
    float compression_ratio() { return compression_ratio_; }
    float average_psnr() { return average_psnr_; }

private:
    float average_compression_time_;
    float average_decompression_time_;
    float compression_ratio_;
    float average_psnr_;
};

InputFile create_input_file(std::string folder_path, std::string filename)
{
    std::ifstream input(folder_path + filename, std::ios::binary);

    if (input.fail())
    {
        // throw std::exception("The filename was invalid.");
        std::cerr << "The filename was invalid." << std::endl;
        std::cout << "goodbit \t" << input.good() << "\t";
        std::cout << "eofbit \t" << input.eof() << "\t";
        std::cout << "failbit \t" << input.bad() << "\t";
        std::cout << "badbit \t" << input.rdstate() << "\t";
    }

    int width;
    int height;
    int byte_size;
    input.read(reinterpret_cast<char *>(&width), sizeof(width));
    input.read(reinterpret_cast<char *>(&height), sizeof(height));
    input.read(reinterpret_cast<char *>(&byte_size), sizeof(byte_size));
    if (byte_size != sizeof(short))
    {
        // throw std::exception("The depth pixels are not 16-bit.");
        std::cerr << "The depth pixels are not 16-bit. - " << byte_size << " bytes" << std::endl;
    }

    return InputFile(filename, std::move(input), width, height);
}

// Converts 16-bit buffers into OpenCV Mats.
cv::Mat create_depth_mat(int width, int height, const short *depth_buffer)
{
    int frame_size = width * height;
    std::vector<char> reduced_depth_frame(frame_size);
    std::vector<char> chroma_frame(frame_size);

    for (int i = 0; i < frame_size; ++i)
    {
        // reduced_depth_frame[i] = depth_buffer[i] / 32;
        reduced_depth_frame[i] = depth_buffer[i];
        // grey for positive delta, purple for negative delta
        if (depth_buffer[i] >= 0)
        {
            chroma_frame[i] = 128;
        }
        else
        {
            chroma_frame[i] = 255;
        }
        // chroma_frame[i] = 128;
    }

    cv::Mat y_channel(height, width, CV_8UC1, reduced_depth_frame.data());
    cv::Mat chroma_channel(height, width, CV_8UC1, chroma_frame.data());

    std::vector<cv::Mat> y_cr_cb_channels;
    y_cr_cb_channels.push_back(y_channel);
    y_cr_cb_channels.push_back(chroma_channel);
    y_cr_cb_channels.push_back(chroma_channel);

    cv::Mat y_cr_cb_frame;
    cv::merge(y_cr_cb_channels, y_cr_cb_frame);

    cv::Mat bgr_frame = y_cr_cb_frame.clone();
    cvtColor(y_cr_cb_frame, bgr_frame, CV_YCrCb2BGR);
    return bgr_frame;
}

float max(std::vector<short> v1)
{
    return *max_element(v1.begin(), v1.end());
}

float mse(std::vector<short> true_values, std::vector<short> encoded_values)
{
    assert(true_values.size() == encoded_values.size());

    int sum = 0;
    int count = 0;
    for (int i = 0; i < true_values.size(); ++i)
    {
        if (true_values[i] == 0)
            continue;
        short error = true_values[i] - encoded_values[i];
        sum += error * error;
        ++count;
    }
    return sum / (float)count;
}

void write_result_output_line(std::ofstream &result_output, InputFile &input_file, std::string type,
                              int change_threshold, int invalidation_threshold, Result result)
{
    result_output << type << ","
                  << input_file.width() << ","
                  << input_file.height() << ","
                  << change_threshold << ","
                  << invalidation_threshold << ","
                  << result.average_compression_time() << ","
                  << result.average_decompression_time() << ","
                  << result.compression_ratio() << ","
                  << result.average_psnr() << std::endl;
}

Result run_rvl(InputFile &input_file)
{
    int frame_size = input_file.width() * input_file.height();
    // std::cout << input_file.width() << std::endl;
    // std::cout << input_file.height() << std::endl;
    int depth_buffer_size = frame_size * sizeof(short);
    // For the raw pixels from the input file.
    std::vector<short> depth_buffer(frame_size);
    // For the RVL compressed frame.
    std::vector<char> rvl_frame;
    // For the decompressed frame.
    std::vector<short> depth_image;

    float compression_time_sum = 0.0f;
    float decompression_time_sum = 0.0f;
    int compressed_size_sum = 0;
    int frame_count = 0;
    while (!input_file.input_stream().eof())
    {
        input_file.input_stream().read(reinterpret_cast<char *>(depth_buffer.data()), depth_buffer_size);

        Timer compression_timer;
        rvl_frame = rvl::compress(depth_buffer.data(), frame_size);
        compression_time_sum += compression_timer.milliseconds();

        Timer decompression_timer;
        depth_image = rvl::decompress(rvl_frame.data(), frame_size);
        decompression_time_sum += decompression_timer.milliseconds();

        auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());

        // char outPath[1024 * 2] = {0};
        // sprintf(outPath, "/home/sc/streamingPipeline/analysisData/trvl/%d.png", frame_count);
        // std::cout << outPath << std::endl;
        // cv::imwrite(outPath, depth_mat);

        // cv::imshow("Depth", depth_mat);
        // if (cv::waitKey(1) >= 0)
        //     break;

        compressed_size_sum += rvl_frame.size();
        // std::cout << "trvl frame size \t" << rvl_frame.size() << "\t";
        ++frame_count;
    }
    input_file.input_stream().close();

    float average_compression_time = compression_time_sum / frame_count;
    float average_decompression_time = decompression_time_sum / frame_count;
    float compression_ratio = (float)(depth_buffer_size * frame_count) / (float)compressed_size_sum;
    std::cout << "RVL" << std::endl
              << "filename: " << input_file.filename() << std::endl
              << "average compression time: " << average_compression_time << " ms" << std::endl
              << "average decompression time: " << average_decompression_time << " ms" << std::endl
              << "compression ratio: " << compression_ratio << std::endl;

    return Result(average_compression_time, average_decompression_time, compression_ratio, 0.0f);
}

// void run_trvl(InputFile &input_file, short change_threshold, int invalidation_threshold, char *outPath)
// {
//     int frame_size = input_file.width() * input_file.height();

//     trvl::Encoder encoder(frame_size, change_threshold, invalidation_threshold);
//     trvl::Decoder decoder(frame_size);

//     std::vector<short> depth_buffer(frame_size);
//     int frame_count = 0;

//     while (!input_file.input_stream().eof())
//     {
//         input_file.input_stream().read(reinterpret_cast<char *>(depth_buffer.data()), frame_size * sizeof(short));

//         bool keyframe = frame_count++ % 30 == 0;
//         auto trvl_frame = encoder.encode(depth_buffer.data(), keyframe);
//         auto depth_image = decoder.decode(trvl_frame.data(), keyframe);
//         auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());

//         // cv::imshow("Depth", depth_mat);
//         cv::imwrite(outPath, depth_mat);
//         std::cout << "writing" << std::endl;
//         // if (cv::waitKey(1) >= 0)
//         //     return;
//     }
// }

template <typename T>
struct Entry
{
    // zeros (_ means indicator)
    bool z_ = false;
    T z = 0;

    // subvec
    bool sv_ = false;
    T d = 0;
    T l = 0;

    // c
    bool c_ = false;
    T c = 0;

    Entry<T>(){};
    Entry<T>(bool z_, T z) : z_(z_), z(z){};
    Entry<T>(bool sv_, T d, T l) : sv_(sv_), d(d), l(l){};
    // Entry<T>(bool c_, T c) : c_(c_), c(c){};
    Entry<T>(bool sv_, T d, T l, bool c_, T c) : sv_(sv_), d(d), l(l), c_(c_), c(c){};
    Entry<T>(bool z_, T z, bool sv_, T d, T l, bool c_, T c) : z_(z_), z(z), sv_(sv_), d(d), l(l), c_(c_), c(c){};
    Entry<T>(const Entry<T> &entry) : z_(entry.z_), z(entry.z), sv_(entry.sv_), d(entry.d), l(entry.l), c_(entry.c_), c(entry.c){};
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Entry<T> &entry)
{
    os << "[z_: " << entry.z_ << ", sv_: " << entry.sv_ << ", c_: " << entry.c_ << "]";
    os << "(" << entry.z << ", " << entry.d << ", " << entry.l << ", " << entry.c << ")";
    return os;
}

template <typename T>
Entry<T> getLargestSubvec(typename std::deque<T>::const_iterator searchL, typename std::deque<T>::const_iterator searchR, typename std::deque<T>::const_iterator lookL, typename std::deque<T>::const_iterator lookR, typename std::deque<T>::const_iterator end)
{
    // std::cout << "search buffer: " << searchBuffer << std::endl;
    // std::cout << "look buffer: " << lookBuffer << std::endl;

    int searchSize = distance(searchL, searchR);

    // start with largest lookBuffer, then pop_back to reduce size of the string to search for
    while (lookL != lookR)
    {
        typename std::deque<T>::const_iterator res = find_end(searchL, searchR, lookL, lookR);
        // std::deque<T>::const_iterator res = search(searchL, searchR, lookL, lookR);
        if (res != searchR)
        {
            // if at end, then do not copy character
            if (lookR == end)
            {
                // TODO: fix problem here with dist from res to lookL
                // sv_ = true, d, l, c_ = false, c
                return Entry<T>(true, distance(res, searchR), distance(lookL, lookR));
            }
            else
            {
                // sv_ = true, d, l, c_ = true, c
                return Entry<T>(true, distance(res, searchR), distance(lookL, lookR), true, *lookR);
            }
        }
        lookR -= 1;
    }
    if (lookL == end)
    {
        // sv_ = false, c_ false
        return Entry<T>();
    }
    else
    {
        // sv_ = false, c_ = true
        return Entry<T>(false, 0, 0, true, *lookL);
    }
}

template <typename T>
int decompLZ77(std::deque<Entry<T>> inBuf, std::deque<T> &out)
{
    // destructive to inBuf
    // read each element and construct the decompressed data
    // std::deque<T> out;
    std::deque<T> searchBuffer;
    // typename std::deque<Entry<T>>::const_iterator itr = inBuf.begin();

    // int size = inBuf.size();
    int count = 0;

    while (inBuf.begin() != inBuf.end())
    {

        Entry<T> entry = inBuf.front();
        // std::cout << entry << std::endl;
        // std::cout << searchBuffer << std::endl;

        // std::cout << entry << std::endl;
        // move back d spaces
        // copy l amount of data

        // searchBuffer.insert(searchBuffer.end(), searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);
        // out.insert(out.end(), searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);

        // appending zeros
        if (entry.z_)
        {
            // by definition cannot have 0 zeros, but double check
            if (entry.z > 0)
            {
                std::deque<T> zerosRun(entry.z, 0);
                out.insert(out.end(), zerosRun.begin(), zerosRun.end());
            }
            else
            {
                std::cout << "zeros is: " << entry.z << " but cannot be <= 0 -> BAD" << std::endl;
            }
        }
        // copying over data while maintaining separate search data
        // TODO: something weird about the bounds of copying to search then to out
        if (entry.sv_)
        {
            std::deque<T> tmp(searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);
            // searchBuffer.insert(searchBuffer.end(), searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);
            // out.insert(out.end(), searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);
            searchBuffer.insert(searchBuffer.end(), tmp.begin(), tmp.end());
            out.insert(out.end(), tmp.begin(), tmp.end());
        }
        if (entry.c_)
        {
            searchBuffer.push_back(entry.c);
            out.push_back(entry.c);
        }
        if (!entry.z_ && !entry.sv_ && !entry.c_)
        {
            std::cout << "z_ = sv_ = c_ = 0 -> BAD uninitialized entry present " << std::endl;
            return -1;
        }
        inBuf.pop_front();
        count++;
    }
    // std::cout << "decompressed:\t\t" << out << std::endl;
    // std::cout << out.max_size() << std::endl;
    // std::cout << out.size() << std::endl;
    return 1;
}

template <typename T>
int check(const std::deque<T> &buf0, const std::deque<T> &buf1)
{
    int ret = 1;
    int count = 0;
    typename std::deque<T>::const_iterator b0 = buf0.begin();
    typename std::deque<T>::const_iterator b1 = buf1.begin();

    if (buf0.size() != buf1.size())
    {
        std::cout << "b0 size: " << buf0.size() << ", b1 size: " << buf1.size() << std::endl;
    }
    while (b0 != buf0.end())
    {
        if (*b0 != *b1)
        {
            std::cout << count << " -> b0: " << *b0 << ", b1: " << *b1 << std::endl;
            ret = 0;
        }
        b0++;
        b1++;
        count++;
    }
    return ret;
}

template <typename T>
int compLZ77(std::deque<T> inBuf, std::deque<Entry<T>> &outBuf)
{
    // reference: https://en.wikipedia.org/wiki/LZ77_and_LZ78#Pseudocode
    int searchWindow = 100;
    int lookWindow = 20;
    int count = 0;

    if (searchWindow > inBuf.size())
    {
        searchWindow = inBuf.size();
    }
    if (lookWindow > searchWindow)
    {
        lookWindow = searchWindow;
    }

    // need to maintain a separate search (virtually) contiguous
    // TODO: fix with rearranging pointers and make sure no data is being copied
    std::deque<T> searchBuffer;

    typename std::deque<T>::const_iterator searchL = searchBuffer.begin();
    typename std::deque<T>::const_iterator searchR = searchBuffer.end();
    typename std::deque<T>::const_iterator lookL = inBuf.begin();
    typename std::deque<T>::const_iterator lookR = inBuf.begin() + lookWindow;
    typename std::deque<T>::const_iterator end = inBuf.end();

    while (lookL != lookR)
    {
        // ensure that search buffer is of the right size
        while (searchBuffer.size() > searchWindow)
        {
            searchBuffer.pop_front();
        }
        // update search buffer bounds
        searchL = searchBuffer.begin();
        searchR = searchBuffer.end();

        // std::cout << "search buffer: " << searchBuffer << std::endl;

        // set lookBuffer
        // std::deque<T> lookBuffer(lookL, lookR);
        // std::cout << "look buffer: " << lookBuffer << std::endl;

        // get run of zeros
        long totalZeros = 0;
        T zeros = 0;
        // while lookL dereferences to 0
        while ((*lookL) == 0)
        {
            // std::cout << "loop " << std::endl;
            if ((lookL != lookR) && (lookL != inBuf.end()) && (lookR != inBuf.end()))
            {
                if (zeros == SHRT_MAX)
                {
                    // only zeros case
                    outBuf.push_back(Entry<T>(true, zeros));
                    totalZeros += zeros;
                    zeros = 0;
                }
                zeros++;
                lookL += 1;
                lookR += 1;
            }
            else
            {
                break;
            }
        }

        // std::cout << zeros << std::endl;
        if ((zeros >= SHRT_MAX) || (zeros < 0))
        {
            std::cout << "zeros invalid: " << zeros << " -> BAD" << std::endl;
        }

        totalZeros += zeros;

        Entry<T> entry = getLargestSubvec<T>(searchL, searchR, lookL, lookR, end);
        if (zeros > 0)
        {
            entry.z_ = true;
            entry.z = zeros;
        }

        outBuf.push_back(entry);
        // std::cout << entry << std::endl;
        // std::cout << searchBuffer.size() << std::endl;

        // add copied data to search buffer and add terminating character
        if (entry.sv_)
        {
            searchBuffer.insert(searchBuffer.end(), searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);
        }
        if (entry.c_)
        {
            searchBuffer.push_back(entry.c);
        }

        // l always 1 at minimum
        int l = entry.l + 1;

        // incrementally update bounds to ensure no overflow
        for (int i = 0; i < l; i++)
        {
            // update bounds of look
            if (lookL != inBuf.end())
            {
                lookL++;
            }
            if (lookR != inBuf.end())
            {
                lookR++;
            }
        }
        count++;
    }

    std::cout << "Compression Finished" << std::endl;

    if (inBuf.size() < 100)
    {
        std::cout << "compressed:\t\t" << outBuf << std::endl;
        // std::cout << "input:\t\t\t" << inBuf << std::endl;
        // std::cout << "decompressed:\t\t" << decompLZ77(outBuf) << std::endl;
    }
    // std::cout << "broken here 0" << std::endl;
    std::deque<T> decomp;
    decompLZ77<T>(outBuf, decomp);
    std::cout << "Decompression Finished" << std::endl;
    // std::cout << "broken here 1" << std::endl;
    int pass = check(inBuf, decomp);
    std::cout << "pass: " << pass << std::endl;

    if (!pass)
    {
        std::cout << "\t original size: " << inBuf.size() << ", uncompressed size: " << decomp.size() << std::endl;
        std::cout << inBuf << std::endl;
        std::cout << decomp << std::endl;
    }

    std::cout << "compression ratio: " << float(sizeof(short) * inBuf.size()) / float((5 * sizeof(T) * outBuf.size()));
    return 1;
}

Result run_trvl(InputFile &input_file, short change_threshold, int invalidation_threshold)
{
    int frame_size = input_file.width() * input_file.height();
    // std::cout << "width \t" << input_file.width() << "\t";
    // std::cout << "height \t" << input_file.height() << "\t";

    int depth_buffer_size = frame_size * sizeof(short);
    std::cout << depth_buffer_size << std::endl;
    // std::cout << "depth buffer size \t" << depth_buffer_size << "\t";
    // For the raw pixels from the input file.
    std::vector<short> depth_buffer(frame_size);
    // For detecting changes and freezing other pixels.
    std::vector<trvl::Pixel> trvl_pixels(frame_size);
    // To save the pixel values of the previous frame to calculate differences between the previous and the current.
    std::vector<short> prev_pixel_values(frame_size);
    // The differences between the adjacent frames.
    std::vector<short> pixel_diffs(frame_size);
    // For the RVL compressed frame.
    std::vector<char> rvl_frame;
    // For the decompressed frame.
    std::vector<short> depth_image;

    float compression_time_sum = 0.0f;
    float decompression_time_sum = 0.0f;
    int compressed_size_sum = 0;
    float psnr_sum = 0.0f;
    int zero_psnr_frame_count = 0;
    int frame_count = 0;

    // std::deque<short> abc = {9, 13, 9, 13, 37, 13, 9, 13, 9, 13, 9, 9};
    // std::deque<short> abc = {1, 20, 30, 30, 20, 10, 2, 3, 4, 5};
    // std::deque<short> abc = {4, 9, 10, 11, 13, 14, 16, 17, 20, 22, 25, 27, 28, 30, 31, 34, 35, 37, 38, 39, 41, 42, 45, 47, 50, 56, 57, 61, 63, 65, 69, 70, 71, 73, 74, 75, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 93, 94, 96, 98, 11,
    //                          4, 9, 10, 11, 13, 14, 16, 17, 20, 22, 25, 27, 28, 30, 31, 34, 35, 37, 38, 39, 41, 42, 45, 47, 50, 56, 57, 61, 63, 65, 69, 70, 71, 73, 74, 75, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 93, 94, 96, 98, 11};
    // // std::deque<short> abc = {4, 9, 10, 11, 13, 14, 16, 17, 20, 22, 25, 27, 28, 30, 31, 3, 7};
    // std::deque<Entry> tmp;
    // compLZ77(abc, tmp);
    // std::cout << abc << std::endl;
    // decompLZ77(tmp);
    frame_count = 0;

    // TODO: why is this reading 101 frames when it should read 100?
    // int total_sum = 0;
    std::ofstream diff_output("/home/sc/streamingPipeline/analysisData/diff.txt");
    // std::ifstream abc("/home/sc/streamingPipeline/analysisData/ref/allDepthBin_1",std::ios::in | std::ios::binary);
    // while ((!input_file.input_stream().eof()))
    // while ((abc.read((char *)(depth_buffer.data()), depth_buffer_size)))
    while ((input_file.input_stream().read((char *)(depth_buffer.data()), depth_buffer_size)) && (frame_count < 5))
    {
        // if (0)
        // {
        // compressing the frame by itself without RVL
        // std::deque<short> abc(depth_buffer.begin(), depth_buffer.end());
        // std::deque<Entry<short>> tmp1;
        // compLZ77(abc, tmp1);

        // do rvl then compress

        // rvl_frame = rvl::compress(depth_buffer.data(), frame_size);

        // for(int i = 0; i < 100; i++){
        //     printf("%d\n",rvl_frame[i]);
        // }

        // std::deque<char> def(rvl_frame.begin(), rvl_frame.end());
        // std::deque<Entry<char>> tmp2;
        // compLZ77(def, tmp2);
        // frame_count++;
        // }
        // else
        // {
        // std::vector<short> abc = {9, 13, 9, 13, 37, 13, 9, 13, 9, 13, 9, 9};
        // std::vector<std::vector<short>> tmp;
        // compLZ77(abc, tmp);
        // for(int i = 1000; i < 1010; i++){
        //     std::cout << depth_buffer[i] << " ";
        // }
        std::cout << frame_count << std::endl;
        // input_file.input_stream().read(reinterpret_cast<char *>(depth_buffer.data()), depth_buffer_size);
        // input_file.input_stream().read(pixel_diffs_char, depth_buffer_size);
        Timer compression_timer;
        // total_sum += depth_buffer.size();
        // std::cout << depth_buffer.size() << std::endl;
        // Update the TRVL pixel values with the raw depth pixels.
        for (int i = 0; i < frame_size; ++i)
        {
            trvl::update_pixel(trvl_pixels[i], depth_buffer[i], change_threshold, invalidation_threshold);
        }

        // std::cout << "depth buffer\t" << depth_buffer_size << "\t";
        // std::cout << "frame size \t" << frame_size << "\t";
        // For the first frame, since there is no previous frame to diff, run vanilla RVL.
        if (frame_count == 0)
        {
            for (int i = 0; i < frame_size; ++i)
            {
                prev_pixel_values[i] = trvl_pixels[i].value;
            }
            rvl_frame = rvl::compress(prev_pixel_values.data(), frame_size);
            // std::cout << "trvl frame size \t" << rvl_frame.size() << "\t";
            compression_time_sum += compression_timer.milliseconds();

            Timer decompression_timer;
            depth_image = rvl::decompress(rvl_frame.data(), frame_size);
            // std::cout << "created depth image " << std::endl;
            decompression_time_sum += decompression_timer.milliseconds();
        }
        else
        {
            // Calculate pixel_diffs using prev_pixel_values
            // and save current pixel values to prev_pixel_values for the next frame.
            for (int i = 0; i < frame_size; ++i)
            {
                short value = trvl_pixels[i].value;
                pixel_diffs[i] = value - prev_pixel_values[i];
                if (pixel_diffs[i] > 65535)
                {
                    std::cout << pixel_diffs[i] << std::endl;
                }
                // if (frame_count == 2)
                // {
                diff_output << pixel_diffs[i] << ",";
                // }
                // std::cout  << value << ",";
                // std::cout << value - prev_pixel_values[i] << " ";

                prev_pixel_values[i] = value;

                // if(pixel_diffs[i] != 0){
                //     std::cout << pixel_diffs[i] << " " ;
                // }
            }
            // std::deque<int> abc;
            // std::deque<Entry<short>> abc;
            // abc.push_back(Entry<short>(0,0,666666666));
            // std::cout << abc.max_size() << std::endl;
            // std::vector<short> tmpVec(pixel_diffs.begin(), pixel_diffs.end());
            std::deque<short> def(pixel_diffs.begin(), pixel_diffs.end());
            // std::cout << def.size() << std::endl;
            std::deque<Entry<short>> tmp2;
            compLZ77(def, tmp2);
            // std::cout << frame_count << std::endl;

            // Compress and decompress the difference.

            // rvl_frame = rvl::compress(pixel_diffs.data(), frame_size);
            rvl_frame = rvl::compress(pixel_diffs.data(), frame_size);
            compression_time_sum += compression_timer.milliseconds();

            Timer decompression_timer;
            auto diff_frame = rvl::decompress(rvl_frame.data(), frame_size);
            // Update depth_image of the previous frame using the difference
            // between the previous frame and the current frame.
            for (int i = 0; i < frame_size; ++i)
            {
                depth_image[i] += diff_frame[i];
            }
            decompression_time_sum += decompression_timer.milliseconds();

            auto depth_mat2 = create_depth_mat(input_file.width(), input_file.height(), pixel_diffs.data());
            char outPath2[1024 * 2] = {0};
            sprintf(outPath2, "/home/sc/streamingPipeline/analysisData/trvl/%d_del.png", frame_count);
            cv::imwrite(outPath2, depth_mat2);
        }

        auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());
        char outPath[1024 * 2] = {0};
        sprintf(outPath, "/home/sc/streamingPipeline/analysisData/trvl/%d_depth.png", frame_count);
        cv::imwrite(outPath, depth_mat);
        // cv::imshow("Depth", depth_mat);
        // cv::imwrite(outPath, depth_mat);
        // if (cv::waitKey(1) >= 0)
        //     break;

        compressed_size_sum += rvl_frame.size();
        // std::cout << "trvl frame size \t" << rvl_frame.size() << "\t";
        // The first frame goes through vanilla RVL which is lossless.
        float mse_value = mse(depth_buffer, depth_image);
        // std::cout << mse_value << std::endl;
        if (mse_value != 0.0f)
        {
            psnr_sum += 20.0f * log10(max(depth_buffer) / sqrt(mse_value));
        }
        else
        {
            ++zero_psnr_frame_count;
        }
        ++frame_count;
        // }
    }
    input_file.input_stream().close();
    diff_output.close();
    // std::cout << total_sum << std::endl;

    float average_compression_time = compression_time_sum / frame_count;
    float average_decompression_time = decompression_time_sum / frame_count;
    float compression_ratio = (depth_buffer_size * frame_count) / (float)compressed_size_sum;
    float average_psnr = (frame_count > zero_psnr_frame_count) ? psnr_sum / (frame_count - zero_psnr_frame_count) : 0.0f;
    std::cout << "Temporal RVL" << std::endl
              << "filename: " << input_file.filename() << std::endl
              << "average compression time: " << average_compression_time << " ms" << std::endl
              << "average decompression time: " << average_decompression_time << " ms" << std::endl
              << "compression ratio: " << compression_ratio << std::endl
              << "average PSNR: " << average_psnr << std::endl;

    return Result(average_compression_time, average_decompression_time, compression_ratio, average_psnr);
}

int main(int argc, char **argv)
{
    if (argc == 4)
    {
        rootPath = argv[1];
        std::cout << rootPath << std::endl;
        inFile = argv[2];
        std::cout << inFile << " ";
        TRVL = std::stoi(argv[3]);
        if (TRVL == 1)
        {
            std::cout << " trvl ";
        }
        else
        {
            std::cout << " rvl ";
        }
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " rootPath binName (0: rvl | 1:trvl)" << std::endl;
        return 0;
    }

    // const std::string RESULT_OUTPUT_FILE_PATH = "/home/sc/streamingPipeline/analysisData/temporal-rvl-data/result.csv";
    // const std::string RESULT_OUTPUT_FILE_PATH = "/home/sc/streamingPipeline/analysisData/temporal-rvl-data/" + std::string(inFile) + "_trvl" + ".csv";

    // char outPath[1024 * 2] = {0};
    // char prefix[1024] = {0};
    // strncpy(prefix, rootPath, strlen(rootPath) - 4);
    // sprintf(outPath, "%strvl/%s.png", prefix, inFile);
    // std::cout << outPath << std::endl;
    // std::cout << rootPath << inFile << std::endl;
    InputFile input_file(create_input_file(rootPath, inFile));

    // if (APPEND == 1)
    // {
    //     std::ofstream result_output(RESULT_OUTPUT_FILE_PATH, std::ios::app);
    if (TRVL == 1)
    {
        // reset_input_file(input_file);
        Result trvl_result(run_trvl(input_file, CHANGE_THRESHOLD, INVALIDATION_THRESHOLD));
    }
    else
    {
        Result rvl_result(run_rvl(input_file));
    }
    // }
    // else
    // {
    //     std::ofstream result_output(RESULT_OUTPUT_FILE_PATH, std::ios::out);
    //     result_output << "trvl,width,height,change_threshold,invalidation_threshold,";
    //     result_output << "average_compression_time,average_decompression_time,";
    //     result_output << "compression_ratio,average_psnr" << std::endl;

    //     if (TRVL == 1)
    //     {
    //         // reset_input_file(input_file);
    //         Result trvl_result(run_trvl(input_file, CHANGE_THRESHOLD, INVALIDATION_THRESHOLD));
    //         write_result_output_line(result_output, input_file, "1", CHANGE_THRESHOLD, INVALIDATION_THRESHOLD, trvl_result);
    //         result_output.close();
    //     }
    //     else
    //     {
    //         Result rvl_result(run_rvl(input_file));
    //         write_result_output_line(result_output, input_file, "0", 0, 0, rvl_result);
    //         result_output.close();
    //     }
    // }
    return 0;
}