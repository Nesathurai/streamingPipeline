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

// void encodeVLE(int &){
//     // assume that the control (size of char, 1 byte) already coded
// };

void EncodeVLE(int value, typename std::deque<int> &pBuffer, int &word, int &nibblesWritten)
{
    do
    {
        int nibble = value & 0x7; // lower 3 bits
        // std::cout << "nib: " << std::bitset<32>(nibble) << std::endl;
        // std::cout << "word0: " << std::bitset<32>(nibble) << std::endl;
        if (value >>= 3)
            nibble |= 0x8; // more to come
        word <<= 4;
        // std::cout << "word1: " << std::bitset<32>(nibble) << std::endl;
        word |= nibble;
        // std::cout << "word2: " << std::bitset<32>(nibble) << std::endl;
        // std::cout << "nibs written: " <<  nibblesWritten << std::endl;
        if (++nibblesWritten == 8)
        { // output word
            pBuffer.push_back(word);
            // pBuffer++;
            // *pBuffer++ = word;
            // std::cout << "word3: " << std::bitset<32>(word) << std::endl;
            // pBuffer.push_back(word);

            nibblesWritten = 0;
            word = 0;
        }
    } while (value);
}

int DecodeVLE(typename std::deque<int> &pBuffer, int &word, int &nibblesWritten)
{
    unsigned int nibble;
    int value = 0, bits = 29;
    do
    {
        if (!nibblesWritten)
        {
            pBuffer.pop_front();
            word = pBuffer.front();
            // word = *pBuffer++; // load word
            nibblesWritten = 8;
        }
        nibble = word & 0xf0000000;
        value |= (nibble << 1) >> bits;
        word <<= 4;
        nibblesWritten--;
        bits -= 3;
    } while (nibble & 0x80000000);
    return value;
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
    // std::cout << "Decompression Finished" << std::endl;
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
int compLZ77(const std::deque<T> &inBuf, std::deque<Entry<T>> &outBuf)
{
    // reference: https://en.wikipedia.org/wiki/LZ77_and_LZ78#Pseudocode
    int searchWindow = 1000;
    int lookWindow = 25;
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

    typename std::deque<T>::const_iterator lookL = inBuf.begin();
    typename std::deque<T>::const_iterator lookR = inBuf.begin() + lookWindow;
    typename std::deque<T>::const_iterator end = inBuf.end();
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

        // std::cout << "search buffer: " << searchBuffer << std::endl;

        // set lookBuffer
        // std::deque<T> lookBuffer(lookL, lookR);
        // std::cout << "look buffer: " << lookBuffer << std::endl;

        // get run of zeros
        // long totalZeros = 0;
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
                    // totalZeros += zeros;
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

        // totalZeros += zeros;

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

    // std::cout << "Compression Finished" << std::endl;
    std::cout << "compression ratio: " << float(sizeof(short) * inBuf.size()) / float((5 * sizeof(T) * outBuf.size())) << std::endl;
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
        return int(ceil(log2(val) / 3) * 4);
    }
}

template <typename T>
int compressLZRVL(std::deque<T> &difDeque, char *output, int numPixels)
{
    // compress with zero runs and LZ77
    // std::deque<short> difDeque(pixel_diffs.begin(), pixel_diffs.end());
    std::deque<Entry<T>> compressed;
    compLZ77(difDeque, compressed);
    // compression_time_sum += compression_timer.milliseconds();

    // serialize data -> // 3 control bits + 4 shorts (8 bytes)

    int originalSize = sizeof(T) * difDeque.size();
    // std::cout << "size buf: " << buffer.size() << " max: " << buffer.max_size() << std::endl;
    std::deque<int> pBuffer;
    // int *pBuffer = (int *)output;
    int word = 0;
    int nibblesWritten = 0;
    // std::deque<T> serialized;
    int count = 0;
    int bitsNeeded = 0;
    // std::bitset<4> bits{"0011"};
    long bits = 0;
    while (compressed.begin() != compressed.end())
    {

        // 2^16 * 2 -> ceil(17/3)*4 = 6*4 = 24 bits = 3 bytes at most
        // 4 * 3 bytes = 12 bytes = 4 ints * 3 + 3 bits
        Entry<T> entry = compressed.front();
        // 3 control = zeros subvec char
        // control will never be negative, so no need for zig zag
        char control = (entry.z_ << 2) + (entry.sv_ << 1) + (entry.c_);
        // std::cout << control << std::endl;

        // if (count == 0)
        // {
        //     std::cout << std::bitset<16>(42) << std::endl;
        //     EncodeVLE(toZigZag(42), pBuffer, word, nibblesWritten);
        //     EncodeVLE(toZigZag(42), pBuffer, word, nibblesWritten);
        //     EncodeVLE(toZigZag(42), pBuffer, word, nibblesWritten);
        //     EncodeVLE(toZigZag(42), pBuffer, word, nibblesWritten);
        //     // std::cout << pBuffer[0] << std::endl;
        //     // std::cout << pBuffer[1] << std::endl;
        //     // std::cout << std::bitset<32>(pBuffer[0]) << std::endl;
        //     // std::cout << std::bitset<32>(pBuffer[1]) << std::endl;
        //     // std::cout << std::bitset<32>(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
        //     // std::cout << std::bitset<32>(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
        //     if (nibblesWritten)
        //     { // last few values
        //         pBuffer.push_back(word << 4 * (8 - nibblesWritten));
        //     }
        //     std::cout << std::bitset<32>(pBuffer[0]) << std::endl;
        // }

        if (entry.z_)
        {
            EncodeVLE(toZigZag(entry.z), pBuffer, word, nibblesWritten); // run of zeros
            if(count == 0){
                std::cout << entry.z << "->" << toZigZag(entry.z) << "->" << bitSize(toZigZag(entry.z)) << std::endl;
            }
            bitsNeeded += bitSize(toZigZag(entry.z));
            // EncodeVLE(int(entry.z), pBuffer, word, nibblesWritten); // run of zeros
        }
        if (entry.sv_)
        {

            EncodeVLE(toZigZag(entry.d), pBuffer, word, nibblesWritten); // distance
            EncodeVLE(toZigZag(entry.l), pBuffer, word, nibblesWritten); // length
            if(count == 0){
                std::cout << entry.d << "->" << toZigZag(entry.d) << "->" << bitSize(toZigZag(entry.d)) << std::endl;
                std::cout << entry.l << "->" << toZigZag(entry.l) << "->" << bitSize(toZigZag(entry.l)) << std::endl;
            }
            bitsNeeded += bitSize(toZigZag(entry.d));
            bitsNeeded += bitSize(toZigZag(entry.l));
            // EncodeVLE(int(entry.d), pBuffer, word, nibblesWritten); // distance
            // EncodeVLE(int(entry.l), pBuffer, word, nibblesWritten); // length
        }
        if (entry.c_)
        {
            // the character (delta) can be between -2^16/2 to 2^16/2
            EncodeVLE(toZigZag(entry.c), pBuffer, word, nibblesWritten); // char
            if(count == 0){
                std::cout << entry.c << "->" << toZigZag(entry.c) << "->" << bitSize(toZigZag(entry.c)) << std::endl;
            }
            bitsNeeded += bitSize(toZigZag(entry.c));
            // EncodeVLE(int(entry.c), pBuffer, word, nibblesWritten); // char
        }
        if (nibblesWritten)
        { // last few values
            pBuffer.push_back(word << 4 * (8 - nibblesWritten));
            // *pBuffer++ = word << 4 * (8 - nibblesWritten);
        }
        if(bitsNeeded > 32){
            std::cout << "bits needed: (+4) " << bitsNeeded << " > 32 -> very BAD" << std::endl;
        }
        if (count == 0)
        {
            // std::cout << "bits needed: (+4) " << bitsNeeded << std::endl;
            // std::cout << entry.z_ << entry.sv_ << entry.c_ << std::endl;
            // std::cout << std::bitset<3>(control) << std::endl;
            // std::cout << entry.z << "->" << toZigZag(entry.z) << "->" << bitSize(toZigZag(entry.z)) << std::endl;
            // std::cout << entry.d << "->" << toZigZag(entry.d) << "->" << bitSize(toZigZag(entry.d)) << std::endl;
            // std::cout << entry.l << "->" << toZigZag(entry.l) << "->" << bitSize(toZigZag(entry.l)) << std::endl;
            // std::cout << entry.c << "->" << toZigZag(entry.c) << "->" << bitSize(toZigZag(entry.c)) << std::endl;
            // std::cout << std::bitset<32>(pBuffer[0]) << std::endl;
            // std::cout << std::bitset<32>(pBuffer[1]) << std::endl;
            // while(pBuffer != buffer.end()){
            // if(*pBuffer != 0){
            //     std::cout << fromZigZag(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // }
            // pBuffer++;
            // }
            // std::cout << "buff size: " << pBuffer.size() << std::endl;
            // std::cout << fromZigZag(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << fromZigZag(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << fromZigZag(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << fromZigZag(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << fromZigZag(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << std::bitset<16>(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << std::bitset<16>(DecodeVLE(pBuffer, word, nibblesWritten)) << std::endl;
            // std::cout << DecodeVLE(pBuffer, word, nibblesWritten) << std::endl;
            // std::cout << DecodeVLE(pBuffer, word, nibblesWritten) << std::endl;
            // std::cout << DecodeVLE(pBuffer, word, nibblesWritten) << std::endl;
        }
        compressed.pop_front();
        count++;
    }
    // pBuffer now contains all serialized data
    // now transfer serialized data to chars

    // std::deque<short> decomp;
    // Timer decompression_timer;
    // decompLZ77<short>(compressed, decomp);
    // decompression_time_sum += decompression_timer.milliseconds();

    // int pass = check(difDeque, decomp);

    // if (!pass)
    // {
    //     std::cout << "pass: " << pass << std::endl;
    //     std::cout << "\t original size: " << difDeque.size() << ", uncompressed size: " << decomp.size() << std::endl;
    //     assert(pass == 1);
    //     // std::cout << inBuf << std::endl;
    //     // std::cout << decomp << std::endl;
    // }

    // int *buffer = (int *)output;
    // int *pBuffer = (int *)output;
    // int word = 0;
    // int nibblesWritten = 0;
    // short *end = input + numPixels;
    // short previous = 0;
    // while (input != end)
    // {
    //     int zeros = 0, nonzeros = 0;
    //     for (; (input != end) && !*input; input++, zeros++);
    //     EncodeVLE(zeros, pBuffer, word, nibblesWritten); // number of zeros
    //     for (short *p = input; (p != end) && *p++; nonzeros++);
    //     EncodeVLE(nonzeros, pBuffer, word, nibblesWritten); // number of nonzeros
    //     for (int i = 0; i < nonzeros; i++)
    //     {
    //         short current = *input++;
    //         int delta = current - previous;
    //         int positive = (delta << 1) ^ (delta >> 31);
    //         EncodeVLE(positive, pBuffer, word, nibblesWritten); // nonzero value
    //         previous = current;
    //     }
    // }

    // if (nibblesWritten) // last few values
    //     *pBuffer++ = word << 4 * (8 - nibblesWritten);

    // return int((char *)pBuffer - (char *)buffer); // num bytes
    return 0;
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

    std::deque<short> all_depth_buffer(frame_size);

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
    while ((input_file.input_stream().read((char *)(depth_buffer.data()), depth_buffer_size)))
    {
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
        // TODO: clean this up? Not my code, don't want to screw up
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
                // TODO: unlikely that there is an overflow, but possible?
                pixel_diffs[i] = value - prev_pixel_values[i];
                diff_output << pixel_diffs[i] << ",";
                prev_pixel_values[i] = value;
            }

            // TODO: speed up this interface -> skip converting to deque
            // the encoding LZRVL
            // Compress the difference.
            std::deque<short> difDeque(pixel_diffs.begin(), pixel_diffs.end());
            std::vector<char> output(5 * frame_size);
            compressLZRVL(difDeque, output.data(), frame_size);
            compression_time_sum += compression_timer.milliseconds();

            // output.resize(size);
            // output.shrink_to_fit();

            // std::deque<short> decomp;
            // Timer decompression_timer;
            // decompLZ77<short>(compressed, decomp);
            // decompression_time_sum += decompression_timer.milliseconds();

            // int pass = check(difDeque, decomp);

            // if (!pass)
            // {
            //     std::cout << "pass: " << pass << std::endl;
            //     std::cout << "\t original size: " << difDeque.size() << ", uncompressed size: " << decomp.size() << std::endl;
            //     assert(pass == 1);
            //     // std::cout << inBuf << std::endl;
            //     // std::cout << decomp << std::endl;
            // }
            // compressed size - using *5 since there are 4 T sized values, plus 3 bits so being conservative
            // compressed_size_sum += float((5 * sizeof(short) * compressed.size()));

            // all_depth_buffer.insert(all_depth_buffer.end(), pixel_diffs.begin(), pixel_diffs.end());

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
    // // the encoding LZRVL
    // // Compress and decompress the difference.
    // std::deque<Entry<short>> compressed;
    // compLZ77(all_depth_buffer, compressed);
    // // compression_time_sum += compression_timer.milliseconds();

    // std::deque<short> decomp;
    // Timer decompression_timer;
    // decompLZ77<short>(compressed, decomp);
    // // decompression_time_sum += decompression_timer.milliseconds();

    // int pass = check(all_depth_buffer, decomp);

    // if (!pass)
    // {
    //     std::cout << "pass: " << pass << std::endl;
    //     std::cout << "\t original size: " << all_depth_buffer.size() << ", uncompressed size: " << decomp.size() << std::endl;
    //     assert(pass == 1);
    //     // std::cout << inBuf << std::endl;
    //     // std::cout << decomp << std::endl;
    // }
    // // compressed size - using *5 since there are 4 T sized values, plus 3 bits so being conservative
    // compressed_size_sum += float((5 * sizeof(short) * compressed.size()));

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