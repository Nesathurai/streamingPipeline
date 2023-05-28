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

        compressed_size_sum += rvl_frame.size();
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

    void reset()
    {
        z_ = false;
        sv_ = false;
        c_ = false;
        z = 0;
        d = 0;
        l = 0;
        c = 0;
    };

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
    int searchWindow = 100;
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
        if (zeros > 0)
        {
            // assert(zeros > 0);
            entry.z_ = true;
            entry.z = zeros;
        }
        assert(entry.z != SHRT_MAX);
        assert(entry.z >= 0);
        assert(entry.d >= 0);
        assert(entry.l >= 0);

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
        assert(l > 0);

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
    }
}

int toSerial(std::vector<bool> &serial, char input)
{
    // only works for control bits -> 0xxx
    // take in the input
    // shift everything to the left
    // write everything to a stream
    std::vector<bool> nibble;
    std::vector<bool>::const_iterator nitr = nibble.begin();
    for (int i = 0; i < 4; ++i)
    {
        nitr = nibble.insert(nitr, ((input & 0xf) & (1 << i)));
    }
    // start of control should only ever be a 1 as the terminating character
    assert(nibble.size() == 4);
    // copy in to serial
    serial.insert(serial.end(), nibble.begin(), nibble.end());

    return 0;
}

int toSerial(std::vector<bool> &serial, int input, int bitsToWrite)
{
    // take in the input
    // shift everything to the left
    // write everything to a stream
    std::vector<bool> nibble;
    int count = 0;
    if (input == 0)
    {
        serial.push_back(0);
        serial.push_back(0);
        serial.push_back(0);
        serial.push_back(0);
        return 0;
    }
    while (bitsToWrite > 0)
    {
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
    assert((nibble.begin() + 1) != nibble.end());
    triple.insert(triple.end(), nibble.begin() + 1, nibble.end());
    assert(triple.size() == 3);
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
    // exit only if at end - no need for terminating char
    if (distance(serial.begin() + count, serial.end()) <= 0)
    {
        return 1;
    }
    // ingest control
    control.insert(control.begin(), serial.begin() + count, serial.begin() + count + 4);
    count += 4;
    bool z_ = entry.z_ = control.at(1);
    bool d_ = entry.sv_ = control.at(2);
    bool l_ = entry.sv_ = control.at(2);
    bool c_ = entry.c_ = control.at(3);
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
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.z = fromZigZag(serialToInt(reassemble));
        z_ = false;
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
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.d = fromZigZag(serialToInt(reassemble));
        assert(entry.d == fromZigZag(serialToInt(reassemble)));
        d_ = false;
        reassemble.clear();
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
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.l = fromZigZag(serialToInt(reassemble));
        assert(entry.l == fromZigZag(serialToInt(reassemble)));
        l_ = false;
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
            reassemble.insert(reassemble.begin(), triple.begin(), triple.end());
            nibble.clear();
            triple.clear();
        } while (cont);
        entry.c = fromZigZag(serialToInt(reassemble));
        assert(entry.c == fromZigZag(serialToInt(reassemble)));
        c_ = false;
        reassemble.clear();
    }
    return 0;
}

template <typename T>
int compressLZRVL(std::deque<T> &pixel_diffs, std::vector<bool> &serialized, int numPixels)
{
    // compress with zero runs and LZ77 (without VLE)
    std::deque<Entry<T>> compressed;
    compLZ77(pixel_diffs, compressed);

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
        // 3 control = zeros subvec char
        // control will never be negative (and will always be 4 bits), so no need for zig zag
        char control = 0x8 + (entry.z_ << 2) + (entry.sv_ << 1) + (entry.c_);
        // std::cout << "control: " << std::bitset<4>(control) << std::endl;
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
        serialized.insert(serialized.end(), serial.begin(), serial.end());
        compressed.pop_front();
        count++;
        entry.reset();
        serial.clear();
    }
    return serialized.size();
}

int decompressLZRVL(std::deque<short> &decompressed, std::vector<bool> &serialized)
{
    // 1st use fromSerial -> compressed
    // 2nd compressed -> deque
    std::deque<Entry<short>> compressed;
    int count = 0;
    Entry<short> entry;
    while (!fromSerial(serialized, count, entry))
    {
        compressed.push_back(entry);
        entry.reset();
    }
    decompLZ77(compressed, decompressed);
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
    while ((input_file.input_stream().read((char *)(depth_buffer.data()), depth_buffer_size)) && (frame_count < 10000))
    {
        if (frame_count % 100 == 0)
        {
            std::cout << "saving frame " << frame_count << std::endl;
        }

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
            // compression_time_sum += compression_timer.milliseconds();
            depth_image = rvl::decompress(rvl_frame.data(), frame_size);
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
            // TODO: compress all at once or each frame individually?
            allPixelDiffs.insert(allPixelDiffs.end(), pixel_diffs.begin(), pixel_diffs.end());
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
    Timer compression_timer;
    compression_time_sum = 0;
    totalSize += compressLZRVL(allPixelDiffs, serialized, frame_size);
    compression_time_sum += compression_timer.milliseconds();

    // compression_time_sum += compression_timer.milliseconds();
    std::deque<short> finalDecompression;
    Timer decompression_timer;
    decompression_time_sum = 0;
    int success = decompressLZRVL(finalDecompression, serialized);
    decompression_time_sum += decompression_timer.milliseconds();

    std::cout << std::endl;
    int pass = check(allPixelDiffs, finalDecompression);
    if (pass)
    {
        std::cout << "LZRVL PASSES" << std::endl;
    }
    else
    {
        std::cout << "LZRVL FAILS" << std::endl;
    }

    input_file.input_stream().close();
    // diff_output.close();

    float average_compression_time = compression_time_sum / frame_count;
    float average_decompression_time = decompression_time_sum / frame_count;
    float compression_ratio = (depth_buffer_size * frame_count) / (float)compressed_size_sum;
    std::cout << "Temporal RVL" << std::endl
              << "filename: " << input_file.filename() << std::endl
              << "average compression time: " << average_compression_time << " ms" << std::endl
              << "average decompression time: " << average_decompression_time << " ms" << std::endl;

    std::cout
        << "diff average compression ratio = in / out = " << float(allPixelDiffs.size() * sizeof(short) * 8) / float(serialized.size()) << std::endl;
    // adding 1 additional frame size for keyframe -> use RVL for 1st frame in future TODO
    std::cout << "overall average compression ratio (includes uncompressed keyframe) = in / out = " << float(frame_count * frame_size * sizeof(short) * 8) / float(frame_size * sizeof(short) * 8 + serialized.size()) << std::endl;
    std::cout << "total uncompressed size (diff) = " << allPixelDiffs.size() << std::endl;
    std::cout << "total uncompressed size (diff + keyframe) = " << allPixelDiffs.size() + frame_size * sizeof(short) * 8 << std::endl;
    std::cout << "total compressed size (diff) = " << serialized.size() << std::endl;
    std::cout << "total compressed size (diff + keyframe) = " << serialized.size() + frame_size * sizeof(short) * 8 << std::endl;

    return Result(average_compression_time, average_decompression_time, compression_ratio, 0);
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