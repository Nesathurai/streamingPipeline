#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <deque>
#include <bitset>
#include <math.h>
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

std::ostream &operator<<(std::ostream &os, const std::vector<bool> &v)
{
    for (int i = 0; i < v.size(); ++i)
    {
        if (i % 4 == 0)
        {
            os << " ";
        };
        os << v.at(i);
    }
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
    Entry<T>(bool sv_, T d, T l, bool c_, T c) : sv_(sv_), d(d), l(l), c_(c_), c(c){};
    Entry<T>(bool z_, T z, bool sv_, T d, T l, bool c_, T c) : z_(z_), z(z), sv_(sv_), d(d), l(l), c_(c_), c(c){};
    Entry<T>(const Entry<T> &entry) : z_(entry.z_), z(entry.z), sv_(entry.sv_), d(entry.d), l(entry.l), c_(entry.c_), c(entry.c){};

    Entry<T> reset() { return Entry(); };

    bool operator==(const Entry<T> &entry)
    {
        bool ret = true;
        if ((z_ != entry.z_) && (sv_ != entry.sv_) && (c_ != entry.c_))
        {
            return false;
        }
        else if ((z != entry.z) && (d != entry.d) && (l != entry.l) && (c != entry.c))
        {
            return false;
        }
        return ret;
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Entry<T> &entry)
{
    os << "[z_: " << entry.z_ << ", sv_: " << entry.sv_ << ", c_: " << entry.c_ << "]";
    os << "(" << entry.z << ", " << entry.d << ", " << entry.l << ", " << entry.c << ")";
    return os;
}

template <typename T>
int toZigZag(T inVal)
{
    // for ints
    return (inVal << 1) ^ (inVal >> 31);
}

template <typename T>
int fromZigZag(T inVal)
{
    // for ints
    return (inVal >> 1) ^ -(inVal & 1);
}

template <typename T>
Entry<T> getLargestSubvec(typename std::deque<T>::const_iterator searchL, typename std::deque<T>::const_iterator searchR, typename std::deque<T>::const_iterator lookL, typename std::deque<T>::const_iterator lookR, typename std::deque<T>::const_iterator end)
{
    int searchSize = distance(searchL, searchR);
    // start with largest lookBuffer, then pop_back to reduce size of the string to search for
    while (lookL != lookR)
    {
        typename std::deque<T>::const_iterator res = find_end(searchL, searchR, lookL, lookR);
        if (res != searchR)
        {
            // if at end, then do not copy character
            if (lookR == end)
            {
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
    std::deque<T> searchBuffer;
    int count = 0;

    while (inBuf.begin() != inBuf.end())
    {
        Entry<T> entry = inBuf.front();
        // if (!entry.z_ && !entry.sv_ && !entry.c_)
        // {
        //     // break;
        //     return 1;
        // }
        // move back d spaces
        // copy l amount of data
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
        if (entry.sv_)
        {
            std::deque<T> tmp(searchBuffer.end() - entry.d, searchBuffer.end() - entry.d + entry.l);
            searchBuffer.insert(searchBuffer.end(), tmp.begin(), tmp.end());
            out.insert(out.end(), tmp.begin(), tmp.end());
        }
        if (entry.c_)
        {
            searchBuffer.push_back(entry.c);
            out.push_back(entry.c);
        }
        // if (!entry.z_ && !entry.sv_ && !entry.c_)
        // {
        //     std::cout << "z_ = sv_ = c_ = 0 -> BAD uninitialized entry present " << std::endl;
        //     return -1;
        // }
        inBuf.pop_front();
        count++;
    }
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
        // return false;
        ret = 0;
    }
    while (b0 != buf0.end())
    {
        if ((b0 != buf0.end()) && b1 != buf1.end())
        {
            if (*b0 != *b1)
            {
                if (count < 5)
                {
                    std::cout << count << " -> b0: " << *b0 << ", b1: " << *b1 << std::endl;
                    count++;
                }
                ret = 0;
            }
            b0++;
            b1++;
        }
        else
        {
            std::cout << "FAIL sizes not equal" << std::endl;
            return ret;
        }
    }
    return ret;
}

template <typename T>
int compLZ77(const std::deque<T> &inputDeque, std::deque<Entry<T>> &outBuf)
{
    // reference: https://en.wikipedia.org/wiki/LZ77_and_LZ78#Pseudocode
    int searchWindow = 1000;
    int lookWindow = 25;
    int count = 0;

    if (searchWindow > inputDeque.size())
    {
        searchWindow = inputDeque.size();
    }
    if (lookWindow > searchWindow)
    {
        lookWindow = searchWindow;
    }

    // need to maintain a separate search (virtually) contiguous
    // TODO: fix with rearranging pointers and make sure no data is being copied
    std::deque<T> searchBuffer;

    typename std::deque<T>::const_iterator lookL = inputDeque.begin();
    typename std::deque<T>::const_iterator lookR = inputDeque.begin() + lookWindow;
    typename std::deque<T>::const_iterator end = inputDeque.end();
    typename std::deque<T>::const_iterator searchL;
    typename std::deque<T>::const_iterator searchR;

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

        // get run of zeros
        T zeros = 0;
        // while lookL dereferences to 0
        while ((*lookL) == 0)
        {
            if ((lookL != lookR) && (lookL != inputDeque.end()) && (lookR != inputDeque.end()))
            {
                // overflow case
                if (zeros == SHRT_MAX / 2 - 1)
                {
                    outBuf.push_back(Entry<T>(true, zeros));
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
        if ((zeros >= SHRT_MAX / 2 - 1) || (zeros < 0))
        {
            std::cout << "zeros invalid: " << zeros << " -> BAD" << std::endl;
        }
        Entry<T> entry = getLargestSubvec<T>(searchL, searchR, lookL, lookR, end);
        assert(zeros != SHRT_MAX);
        if (zeros > 0)
        {
            entry.z_ = true;
            entry.z = zeros;
        }

        outBuf.push_back(entry);
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
            if (lookL != inputDeque.end())
            {
                lookL++;
            }
            if (lookR != inputDeque.end())
            {
                lookR++;
            }
        }
        count++;
    }

    std::cout << "compression ratio (without VLE): " << float(sizeof(short) * inputDeque.size()) / float((5 * sizeof(T) * outBuf.size())) << std::endl;
    return 1;
}

int bitSize(int val)
{
    if (val == 0)
    {
        return 0; // empty value will not be saved
    }
    else
    {
        // eg: 1000 -> 000 100
        return int(ceil(log2(val + 1) / 3) * 4);
        // return int(ceil(log2(val)))*8;
    }
}

int toSerial(std::vector<bool> &serial, char input)
{
    // only works for control bits -> 0xxx
    // take in the input
    // shift everything to the left
    // write everything to a stream
    std::vector<bool> nibble;
    // nibble.reserve(4);
    std::vector<bool>::const_iterator nitr = nibble.begin();
    for (int i = 0; i < 4; ++i)
    {
        nitr = nibble.insert(nitr, ((input & 0xf) & (1 << i)));
    }
    // start of control should only ever be a 1 as the terminating character
    // assert(nibble.at(0) == 0);
    assert(nibble.size() == 4);
    // copy in to serial
    // serial.reserve(serial.size() + nibble.size());
    serial.insert(serial.end(), nibble.begin(), nibble.end());
    // std::cout << nibble << std::endl;
    // nibble.clear();
    return 0;
}

int toSerial(std::vector<bool> &serial, int input, int bitsToWrite)
{
    // take in the input
    // shift everything to the left
    // write everything to a stream
    std::vector<bool> nibble;
    // std::vector<bool>::const_iterator nitr = nibble.begin();
    int count = 0;

    // add additional padding if needed (up to 2 0s)
    // int padding = bitsToWrite % 3;
    // bitsToWrite += padding;
    // bitsToWrite += padding;
    // int remainder = 3-padding;
    // for(int i = 0; i < padding; ++i){
    //     nibble.emplace(nibble.end(), 0);
    // }
    // check
    // std::cout << input << std::endl;
    // bitsToWrite = bitsToWrite + bitsToWrite%3;
    std::cout << input << " " << std::bitset<16>(input) << std::endl;
    while (bitsToWrite > 0)
    {
        std::cout << "btw: " << bitsToWrite << " ";

        if (bitsToWrite <= 3)
        {
            nibble.insert(nibble.end(), 0);
        }
        else
        {
            nibble.insert(nibble.end(), 1);
        }
        for (int i = 2; i >= 0; --i)
        {
            nibble.insert(nibble.end(), ((input & 0x07) & (1 << i)));
        }
        std::cout << "nib: " << nibble << std::endl;
        bitsToWrite -= 3;
        input >>= 3;
        count++;
    }

    serial.insert(serial.end(), nibble.begin(), nibble.end());

    return 0;
}

int fromSerialToTriple(std::vector<bool> &nibble, std::vector<bool> &triple)
{
    assert(nibble.size() == 4);

    triple.insert(triple.end(), nibble.begin() + 1, nibble.end());
    // // val <<= 1;
    // val += nibble.at(1);
    // // val <<= 1;
    // val += nibble.at(2);
    // // val <<= 1;
    // val += nibble.at(3);

    // val <<= 3; // shift val
    // val += (nibble.at(1) << 2);
    // val += (nibble.at(2) << 1);
    // val += (nibble.at(3) << 0);
    // std::cout << std::bitset<16>(val) << std::endl;

    // val += (nibble.at(1) << 2);
    // val += (nibble.at(2) << 1);
    // val += (nibble.at(3) << 0);
    // TODO: val should be here?
    // val <<= 3; // shift val
    return nibble.at(0);
}

int serialToInt(std::vector<bool> &serial)
{
    int val = 0;
    for (int i = 0; i < serial.size(); i++)
    {
        val <<= 1;
        val += serial[i];
    }
    return val;
}

int fromSerial(std::vector<bool> &serial, int &count, Entry<short> &entry)
{
    // decode serial data - each run should contain the control, then optionally z, sv, c
    std::vector<bool> control;
    // control.reserve(4);
    // Entry<short> entry;
    // int count = 0;
    // for (int i = 0; i < 20; i += 4)
    // {
    //     if(i == 0){
    //         std::cout << "control: " << std::endl;
    //     }
    //     std::cout << serial.at(i) << serial.at(i + 1) << serial.at(i + 2) << serial.at(i + 3) << std::endl;
    // }
    // std::cout << "ser size" << serial.size() << std::endl;

    // exit only if at end - no need for terminating char
    std::cout << "dist: " << distance(serial.begin() + count, serial.end()) << std::endl;
    if (distance(serial.begin() + count, serial.end()) <= 0)
    {
        std::cout << "end of serial" << std::endl;
        return 1;
    }
    // ingest control
    control.insert(control.begin(), serial.begin() + count, serial.begin() + count + 4);
    // only increment if it is not at the end of a run / entry?
    count += 4;

    // check that bit 0 is always 0 except when the term char
    int z_ = entry.z_ = control.at(1);
    int d_ = entry.sv_ = control.at(2);
    int l_ = entry.sv_ = control.at(2);
    int c_ = entry.c_ = control.at(3);
    // if ((control.at(0) == 0) && (control.at(1) == 0) && (control.at(2) == 0) && (control.at(3) == 0))
    // {
    //     std::cout << control << std::endl;
    //     // TODO: shouldnt have to do this, but as redundancy
    //     // NOTE: make a better special case?
    //     entry.z_ = 0;
    //     entry.sv_ = 0;
    //     entry.c_ = 0;
    //     entry.z = -1;
    //     entry.d = -1;
    //     entry.l = -1;
    //     entry.c = -1;
    //     std::cout << "decompressing finished" << std::endl;
    //     // control.clear();
    //     // return entry;
    //     return 1;
    // }

    if (z_)
    {
        std::vector<bool> reassemble;
        int cont = 0;
        do
        {
            std::vector<bool> nibble;
            std::vector<bool> triple;
            nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
            count += 4;
            cont = fromSerialToTriple(nibble, triple);
            std::cout << nibble << std::endl;
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.z = fromZigZag(serialToInt(reassemble));
        std::cout << "z: " << entry.z << std::endl;
        z_ = 0;
        reassemble.clear();
        assert(entry.z > 0);
    }
    if (d_)
    {
        std::vector<bool> reassemble;
        int cont = 0;
        do
        {
            std::vector<bool> nibble;
            std::vector<bool> triple;
            nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
            count += 4;
            cont = fromSerialToTriple(nibble, triple);
            std::cout << nibble << std::endl;
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        std::cout << reassemble << std::endl;
        entry.d = fromZigZag(serialToInt(reassemble));
        std::cout << "d: " << entry.d << std::endl;
        d_ = 0;
        reassemble.clear();
        std::cout << entry << std::endl;

        assert(entry.d > 0);
    }
    if (l_)
    {
        std::vector<bool> reassemble;
        int cont = 0;
        do
        {
            std::vector<bool> nibble;
            std::vector<bool> triple;
            nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
            count += 4;
            cont = fromSerialToTriple(nibble, triple);
            std::cout << nibble << std::endl;
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.l = fromZigZag(serialToInt(reassemble));
        std::cout << "l: " << entry.l << std::endl;
        l_ = 0;
        reassemble.clear();
        assert(entry.l > 0);
    }
    if (c_)
    {
        std::vector<bool> reassemble;
        int cont = 0;
        do
        {
            std::vector<bool> nibble;
            std::vector<bool> triple;
            nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
            count += 4;

            cont = fromSerialToTriple(nibble, triple);
            std::cout << nibble << std::endl;
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.c = fromZigZag(serialToInt(reassemble));
        std::cout << "c: " << entry.c << std::endl;
        c_ = 0;
        reassemble.clear();
    }
    // int cont = 0;
    // while (z_)
    // {
    //     std::vector<bool> reassemble;
    //     std::vector<bool> nibble;
    //     std::vector<bool> triple;
    //     nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
    //     count += 4;

    //     // need to insert this left to right
    //     cont = fromSerialToTriple(nibble, triple);
    //     // std::cout << nibble << std::endl;
    //     reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
    //     if (!cont)
    //     {
    //         entry.z = fromZigZag(serialToInt(reassemble));
    //         std::cout << "z: " << entry.z << std::endl;
    //         z_ = 0;
    //         reassemble.clear();
    //     }
    // }
    // while (d_)
    // {
    //     std::vector<bool> reassemble;
    //     std::vector<bool> nibble;
    //     std::vector<bool> triple;
    //     nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
    //     count += 4;

    //     // need to insert this left to right
    //     cont = fromSerialToTriple(nibble, triple);
    //     reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
    //     if (!cont)
    //     {
    //         entry.d = fromZigZag(serialToInt(reassemble));
    //         std::cout << "d: " << entry.d << std::endl;
    //         d_ = 0;
    //     }
    // }
    // while (l_)
    // {
    //     std::vector<bool> reassemble;
    //     std::vector<bool> nibble;
    //     std::vector<bool> triple;
    //     nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
    //     count += 4;

    //     // need to insert this left to right
    //     cont = fromSerialToTriple(nibble, triple);
    //     reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
    //     if (!cont)
    //     {
    //         entry.l = fromZigZag(serialToInt(reassemble));
    //         std::cout << "l: " << entry.l << std::endl;
    //         l_ = 0;
    //     }
    // }
    // while (c_)
    // {
    //     std::vector<bool> reassemble;
    //     std::vector<bool> nibble;
    //     std::vector<bool> triple;
    //     nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
    //     count += 4;

    //     // need to insert this left to right
    //     cont = fromSerialToTriple(nibble, triple);
    //     // std::cout << nibble << std::endl;
    //     reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
    //     if (!cont)
    //     {
    //         entry.c = fromZigZag(serialToInt(reassemble));
    //         std::cout << "c: " << entry.c << std::endl;
    //         c_ = 0;
    //         reassemble.clear();
    //     }
    // }

    // while (z_ || d_ || l_ || c_)
    // {
    //     std::cout << "cntl " << control.at(0) << control.at(1) << control.at(2) << control.at(3) << " -> ";
    //     std::cout << "z_ d_ l_ c_ : " << z_ << d_ << l_ << c_ << std::endl;
    //     std::vector<bool> reassemble;
    //     int val = 0;
    //     int cont = 0;

    //     // ingest nibble (4 bits)
    //     do
    //     {

    //         std::vector<bool> nibble;
    //         std::vector<bool> triple;
    //         nibble.insert(nibble.end(), serial.begin() + count, serial.begin() + count + 4);
    //         // problem here with overflow
    //         // std::cout << nibble << std::endl;
    //         count += 4;

    //         // see if you should keep going
    //         // val <<= 3; // shift val
    //         // need to insert this left to right
    //         cont = fromSerialToTriple(nibble, triple);
    //         std::cout << nibble << std::endl;
    //         reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
    //         if (d_)
    //         {
    //             // d_= 0;
    //             cont = 1;
    //             entry.d = fromZigZag(serialToInt(reassemble));
    //             std::cout << "d: " << entry.d << std::endl;
    //             d_ = 0;
    //             reassemble.clear();
    //         }
    //         // std::cout << reassemble << std::endl;
    //         // std::cout << "v: " << val << "-> " << std::bitset<16>(val) << std::endl;
    //         // now get data starting with z, then sv, then c
    //         nibble.clear();
    //         triple.clear();
    //         // count++;
    //     } while (cont);
    //     if (z_)
    //     {
    //         entry.z = fromZigZag(serialToInt(reassemble));
    //         std::cout << "z: " << entry.z << std::endl;
    //         z_ = 0;
    //     }
    //     else if (d_)
    //     {
    //         entry.d = fromZigZag(serialToInt(reassemble));
    //         std::cout << "d: " << entry.d << std::endl;
    //         d_ = 0;
    //     }
    //     else if (l_)
    //     {
    //         entry.l = fromZigZag(serialToInt(reassemble));
    //         std::cout << "l: " << entry.l << std::endl;
    //         l_ = 0;
    //     }
    //     else if (c_)
    //     {
    //         std::cout << reassemble << std::endl;
    //         entry.c = fromZigZag(serialToInt(reassemble));
    //         std::cout << "c: " << entry.c << std::endl;
    //         c_ = 0;
    //     }
    //     reassemble.clear();
    // }
    std::cout << "return fromSerial" << std::endl;
    return 0;
}

template <typename T>
int compressLZRVL(std::deque<T> &pixel_diffs, std::vector<bool> &serialized, int numPixels)
{
    // compress with zero runs and LZ77 (without VLE)
    std::deque<Entry<T>> compressed;
    compLZ77(pixel_diffs, compressed);
    for (int i = 0; i < compressed.size(); i++)
    {
        std::cout << compressed[i] << " ";
    }
    std::cout << std::endl;

    // compression_time_sum += compression_timer.milliseconds();
    // serialize data -> // 3 control bits + 4 shorts (8 bytes)
    // std::cout << "ser pixel diffs: " << pixel_diffs.size() << std::endl;
    // std::cout << "ser compressed: " << compressed.size() << std::endl;
    // std::cout << "ser size comp: " << serialized.size() << std::endl;
    // int originalSize = sizeof(T) * pixel_diffs.size();
    int count = 0;
    int totalBits = 0;

    // now allocate the output
    std::vector<bool> serial;
    // serial.reserve(totalBits + 4); // extra 4 bits for terminating character

    while (compressed.begin() != compressed.end())
    {
        // std::cout << count << std::endl;
        // 2^16 * 2 -> ceil(17/3)*4 = 6*4 = 24 bits = 3 bytes at most
        // 4 * 3 bytes = 12 bytes = 4 ints * 3 + 3 bits
        Entry<T> entry = compressed.front();
        std::cout << entry << std::endl;
        // 3 control = zeros subvec char
        // control will never be negative, so no need for zig zag
        char control = 0x8 + (entry.z_ << 2) + (entry.sv_ << 1) + (entry.c_);
        std::cout << "control: " << std::bitset<4>(control) << std::endl;
        toSerial(serial, control);
        // std::cout << "ser tmp serial: " << serial.size() << std::endl;
        if (entry.z_)
        {
            assert(entry.z > 0);
            int z = toZigZag(entry.z);
            toSerial(serial, z, ceil(log2(z + 1))); // run of zeros
        }
        if (entry.sv_)
        {
            assert(entry.d > 0);
            assert(entry.l > 0);
            int d = toZigZag(entry.d);
            int l = toZigZag(entry.l);
            // +1 to deal with weird edge case eg. 8 -> ceil(log2(8)) = 3, but actually needs 4 -> 1000
            toSerial(serial, d, ceil(log2(d + 1))); // distance
            toSerial(serial, l, ceil(log2(l + 1))); // length
        }
        if (entry.c_)
        {
            int c = toZigZag(entry.c);
            // the character (delta) can be between -2^16/2 to 2^16/2
            toSerial(serial, c, ceil(log2(c + 1))); // char
        }
        // std::cout << "ser serial: " << serialized.size() << std::endl;
        // serialized.resize(serialized.size() + serial.size());
        // std::cout << "ser size: " << serial.size() << std::endl;
        // if(count == 0){
        std::cout << "entry " << entry << "-> " << serial << std::endl;
        serialized.insert(serialized.end(), serial.begin(), serial.end());
        compressed.pop_front();
        count++;
        // std::cout << "count: " << count << " -> " << entry << " -> ";
        // std::cout << serial << std::endl;
        entry.reset();
        serial.clear();
    }

    // serialized.insert(serialized.end(), serial.begin(), serial.end());
    // serial.clear();
    // now that the whole file has been encoded, add terminating character ie cntl = 1000

    // char termChar = 0x8;
    // toSerial(serialized, termChar);
    // serialized.push_back(0);
    // serialized.push_back(0);
    // serialized.push_back(0);
    // serialized.push_back(0);

    std::cout << " adding terminating character " << std::endl;
    std::cout << serialized.at(serialized.size() - 5) << serialized.at(serialized.size() - 4) << serialized.at(serialized.size() - 3) << serialized.at(serialized.size() - 2) << serialized.at(serialized.size() - 1) << std::endl;
    // assert(serialized.at(serialized.size() - 4) == 1);
    // totalBits + 4 should be equal to serialized.size()
    // std::cout << "ser: " << serialized.size() << " total bits+4 = " << totalBits+4 << std::endl;
    // assert(serialized.size() == (totalBits + 4));
    return serialized.size();
}

int decompressLZRVL(std::deque<short> &decompressed, std::vector<bool> &serialized)
{
    // 1st use fromSerial -> compressed
    // 2nd compressed -> deque
    // std::vector<bool>::const_iterator sitr = serialized.begin();
    std::deque<Entry<short>> compressed;
    int count = 0;
    Entry<short> entry;
    while (!fromSerial(serialized, count, entry))
    {
        std::cout << "count: " << count << std::endl;
        compressed.push_back(entry);
        entry.reset();
    }
    // std::cout << "count: " << count << std::endl;
    // for (int i = 0; i < compressed.size(); i++)
    // {
    //     std::cout << compressed[i] << std::endl;
    // }
    // while (true)
    // {
    //     Entry<short> entry;
    //     // sitr = serialized.begin() + count;
    //     int end = fromSerial(serialized, count, entry);
    //     // sitr += 8;
    //     // count += 8;
    //     if (count > serialized.size())
    //     {
    //         std::cout << "inf loop -> BAD EXIT " << std::endl;
    //         break;
    //     }
    //     if (end == 1)
    //     {
    //         std::cout << "end " << std::endl;
    //         break;
    //     }
    //     else if (end == 0)
    //     {
    //         std::cout << "count " << count << std::endl;
    //         compressed.push_back(entry);
    //     }
    //     else
    //     {
    //         std::cout << "end was not 0 or 1: " << end << " -> very BAD" << std::endl;
    //         assert(0);
    //     }
    //     // entry.clear();
    // }
    std::cout << "total compressed elements: " << compressed.size() << std::endl;
    std::cout << decompLZ77(compressed, decompressed) << std::endl;

    return 1;
}

Result run_trvl(InputFile &input_file, short change_threshold, int invalidation_threshold)
{
    int frame_size = input_file.width() * input_file.height();
    // std::cout << "width \t" << input_file.width() << "\t";
    // std::cout << "height \t" << input_file.height() << "\t";

    int depth_buffer_size = frame_size * sizeof(short);
    // For the raw pixels from the input file.
    std::vector<short> depth_buffer(frame_size);
    // For detecting changes and freezing other pixels.
    std::vector<trvl::Pixel> trvl_pixels(frame_size);
    // To save the pixel values of the previous frame to calculate differences between the previous and the current.
    std::vector<short> prev_pixel_values(frame_size);
    // The differences between the adjacent frames.
    std::deque<short> pixel_diffs(frame_size);
    // For the RVL compressed frame.
    std::vector<char> rvl_frame;
    // For the decompressed frame.
    std::vector<short> depth_image;
    std::deque<short> all_depth_buffer(frame_size);

    float compression_time_sum = 0.0f;
    float decompression_time_sum = 0.0f;
    int compressed_size_sum = 0;
    float psnr_sum = 0.0f;
    int zero_psnr_frame_count = 0;
    int frame_count = 0;
    int totalSize = 0; // in bits
    std::vector<bool> serialized;
    std::deque<short> allPixelDiffs;
    // TODO: why is this reading 101 frames when it should read 100?
    // std::ofstream diff_output("/home/sc/streamingPipeline/analysisData/diff.txt");
    // while ((!input_file.input_stream().eof()))
    while ((input_file.input_stream().read((char *)(depth_buffer.data()), depth_buffer_size)) && (frame_count < 2))
    {
        std::cout << frame_count << std::endl;
        Timer compression_timer;

        // Update the TRVL pixel values with the raw depth pixels.
        for (int i = 0; i < frame_size; ++i)
        {
            trvl::update_pixel(trvl_pixels[i], depth_buffer[i], change_threshold, invalidation_threshold);
        }

        // For the first frame, since there is no previous frame to diff, run vanilla RVL.
        // TODO: clean this up? Not my code, don't want to screw up
        if (frame_count == 0)
        {
            for (int i = 0; i < frame_size; ++i)
            {
                prev_pixel_values[i] = trvl_pixels[i].value;
            }
            rvl_frame = rvl::compress(prev_pixel_values.data(), frame_size);
            compression_time_sum += compression_timer.milliseconds();
            Timer decompression_timer;
            depth_image = rvl::decompress(rvl_frame.data(), frame_size);
            decompression_time_sum += decompression_timer.milliseconds();
        }
        else
        {
            // Calculate pixel_diffs using prev_pixel_values
            // and save current pixel values to prev_pixel_values for the next frame.
            for (int i = 0; i < frame_size; ++i)
            {
                short value = trvl_pixels[i].value;
                // TODO: unlikely that there is an overflow, but possible?
                pixel_diffs[i] = value - prev_pixel_values[i];
                // diff_output << pixel_diffs[i] << ",";
                prev_pixel_values[i] = value;
            }

            // the encoding LZRVL
            // Compress the difference.
            // TODO: need to fix?
            allPixelDiffs.insert(allPixelDiffs.end(), pixel_diffs.begin(), pixel_diffs.end());
            // TODO: this reserve is wrong??? mem leak
            // serialized.reserve(serialized.size() + frame_size * 5);
            // totalSize += compressLZRVL(pixel_diffs, serialized, frame_size);
            // // std::cout << serialized.size() << std::endl;
            // compression_time_sum += compression_timer.milliseconds();

            // auto depth_mat2 = create_depth_mat(input_file.width(), input_file.height(), pixel_diffs.data());
            // char outPath2[1024 * 2] = {0};
            // sprintf(outPath2, "/home/sc/streamingPipeline/analysisData/trvl/%d_del.png", frame_count);
            // cv::imwrite(outPath2, depth_mat2);
        }

        // auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());
        // char outPath[1024 * 2] = {0};
        // sprintf(outPath, "/home/sc/streamingPipeline/analysisData/trvl/%d_depth.png", frame_count);
        // cv::imwrite(outPath, depth_mat);
        ++frame_count;
    }

    // std::vector<Entry<short>> abc;
    // Entry<short> tmp;
    // tmp.z_ = 1;
    // tmp.z = 20240;
    // tmp.sv_ = 1;
    // tmp.d = 69;
    // tmp.l = 420;
    // tmp.c_ = 1;
    // tmp.c = 47;
    // abc.push_back(tmp);
    // std::vector<bool> serial;
    // // 3 control = zeros subvec char
    // // control will never be negative, so no need for zig zag
    // char control = 0x8 + (tmp.z_ << 2) + (tmp.sv_ << 1) + (tmp.c_);
    // toSerial(serial, control);

    // // std::cout << "ser tmp serial: " << serial.size() << std::endl;
    // if (tmp.z_)
    // {
    //     int z = toZigZag(tmp.z);
    //     std::cout << "z: " << z << std::endl;
    //     toSerial(serial, z, ceil(log2(z))); // run of zeros
    // }
    // if (tmp.sv_)
    // {
    //     int d = toZigZag(tmp.d);
    //     int l = toZigZag(tmp.l);
    //     toSerial(serial, d, ceil(log2(d))); // distance
    //     toSerial(serial, l, ceil(log2(l))); // length
    // }
    // if (tmp.c_)
    // {
    //     int c = toZigZag(tmp.c);
    //     // the character (delta) can be between -2^16/2 to 2^16/2
    //     toSerial(serial, c, ceil(log2(c))); // char
    // }

    // control = 0x8 + (tmp.z_ << 2) + (tmp.sv_ << 1) + (tmp.c_);
    // toSerial(serial, control);

    // // std::cout << "ser tmp serial: " << serial.size() << std::endl;
    // if (tmp.z_)
    // {
    //     int z = toZigZag(tmp.z);
    //     std::cout << "z: " << z << std::endl;
    //     toSerial(serial, z, ceil(log2(z))); // run of zeros
    // }
    // if (tmp.sv_)
    // {
    //     int d = toZigZag(tmp.d);
    //     int l = toZigZag(tmp.l);
    //     toSerial(serial, d, ceil(log2(d))); // distance
    //     toSerial(serial, l, ceil(log2(l))); // length
    // }
    // if (tmp.c_)
    // {
    //     int c = toZigZag(tmp.c);
    //     // the character (delta) can be between -2^16/2 to 2^16/2
    //     toSerial(serial, c, ceil(log2(c))); // char
    // }

    // // adding term char
    // // int term = 0x8;
    // serial.push_back(1);
    // serial.push_back(0);
    // serial.push_back(0);
    // serial.push_back(0);
    // std::cout << "serial: " << serial << std::endl;

    // int cnt = 0;
    // Entry<short> ot;
    // int end = fromSerial(serial, cnt, ot);
    // std::cout << "end: " << end << " out: " << ot << std::endl;
    // end = fromSerial(serial, cnt, ot);
    // std::cout << "end: " << end << " out: " << ot << std::endl;
    // end = fromSerial(serial, cnt, ot);
    // if (end)
    // {
    //     std::cout << "decompression completed!!!" << std::endl;
    // }

    std::deque<short> a;
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(6940);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(4321);
    a.push_back(10030);
    a.push_back(17000);
    a.push_back(10000);
    a.push_back(19000);
    a.push_back(15000);
    a.push_back(14010);
    a.push_back(19870);
    a.push_back(0);
    a.push_back(0);
    a.push_back(0);
    a.push_back(0);
    a.push_back(0);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(99);
    a.push_back(99);
    a.push_back(99);
    a.push_back(99);
    a.push_back(99);
    a.push_back(99);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10100);
    a.push_back(1);
    a.push_back(1);
    a.push_back(1);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(10000);
    a.push_back(33);
    a.push_back(31415);
    totalSize += compressLZRVL(allPixelDiffs, serialized, frame_size);
    // totalSize += compressLZRVL(a, serialized, frame_size);
    // std::cout << serialized << std::endl;
    // int cnt = 0;
    // Entry<short> ot;
    // int end = fromSerial(serialized, cnt, ot);
    // if (end)
    // {
    //     std::cout << "decompression completed!!!" << std::endl;
    // }

    // for (int i = 0; i < 32; i += 8)
    // {
    //     if (i == 0)
    //     {
    //         std::cout << "control: " << std::endl;
    //     }
    //     std::cout << serialized.at(i) << serialized.at(i + 1) << serialized.at(i + 2) << serialized.at(i + 3) << " ";
    //     std::cout << serialized.at(i + 4) << serialized.at(i + 5) << serialized.at(i + 6) << serialized.at(i + 7) << std::endl;
    // }

    // std::cout << serialized.size() << std::endl;
    // compression_time_sum += compression_timer.milliseconds();
    std::cout << "ser size after comp : " << serialized.size() << std::endl;
    std::deque<short> finalDecompression;
    int success = decompressLZRVL(finalDecompression, serialized);
    std::cout << "ser size after decomp : " << serialized.size() << std::endl;

    // TODO: why am I getting 2 different sizes?
    // std::cout << "average compression ratio = in / out = " << float(frame_size * frame_count * sizeof(short) * 8) / float(totalSize) << std::endl;
    // std::cout << "serialized size: " << serialized.size() << " totalBits: " << totalSize << " capacity: " << serialOut.capacity()<< " reserved: " <<5 * frame_size * 8 << std::endl;
    // std::cout << "diff average compression ratio = in / out = " << float(frame_size * frame_count * sizeof(short) * 8) / float(serialized.size()) << std::endl;
    // std::cout << "overall average compression ratio = in / out = " << float(allPixelDiffs.size() * sizeof(short) * 8) / float(serialized.size()) << std::endl;
    // adding 1 additional frame size for keyframe
    // TODO: NOTE: this is with completely uncompressed keyframe (in future do rvl)

    // std::cout << "diff average compression ratio = in / out = " << float(allPixelDiffs.size() * sizeof(short) * 8) / float(serialized.size()) << std::endl;
    // std::cout << "overall average compression ratio = in / out = " << float(frame_count * frame_size * sizeof(short) * 8) / float(frame_size * sizeof(short) * 8 + serialized.size()) << std::endl;
    // std::cout << finalDecompression.size() << std::endl;
    // std::cout << "total uncompressed diff size = " << allPixelDiffs.size() << std::endl;
    // std::cout << "total uncompressed size (diff + key) = " << allPixelDiffs.size() + frame_size * 8 * sizeof(short) << std::endl;

    int pass = check(allPixelDiffs, finalDecompression);
    // int pass = check(a, finalDecompression);
    std::cout << "pixel diffs pass: " << pass << std::endl;

    std::cout << a << std::endl;

    // std::cout << compressed << std::endl;
    std::cout << finalDecompression << std::endl;

    input_file.input_stream().close();
    // diff_output.close();

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