#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
// #include <iostream>
#include "trvl.h"

short CHANGE_THRESHOLD = 10;
int INVALIDATION_THRESHOLD = 2;
int TRVL = 1;
int APPEND = 0;
int width = 640 * 2;
int height = 576 * 2;
char *inFile;
char *rootPath;

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
    // std::ifstream input("/home/sc/streamingPipeline/analysisData/temporal-rvl-data/comm-camera-movement", std::ios::binary);
    // std::ifstream input("/home/sc/streamingPipeline/analysisData/temporal-rvl-data/bin", std::ios::binary);

    if (input.fail())
    {
        // throw std::exception("The filename was invalid.");
        std::cerr << "The filename was invalid." << std::endl;
    }

    // int width;
    // int height;
    // int byte_size;
    // input.read(reinterpret_cast<char *>(&width), sizeof(width));
    // input.read(reinterpret_cast<char *>(&height), sizeof(height));
    // input.read(reinterpret_cast<char *>(&byte_size), sizeof(byte_size));
    // if (byte_size != sizeof(short))
    // {
    //     // throw std::exception("The depth pixels are not 16-bit.");
    //     std::cerr << "The depth pixels are not 16-bit. - " << byte_size << " bytes" << std::endl;
    // }

    return InputFile(filename, std::move(input), width, height);
    // return InputFile("/home/sc/streamingPipeline/analysisData/temporal-rvl-data/bin", std::move(input), width, height);
}

// Converts 16-bit buffers into OpenCV Mats.
cv::Mat create_depth_mat(int width, int height, const short *depth_buffer)
{
    int frame_size = width * height;
    std::vector<char> reduced_depth_frame(frame_size);
    std::vector<char> chroma_frame(frame_size);

    for (int i = 0; i < frame_size; ++i)
    {
        reduced_depth_frame[i] = depth_buffer[i] / 32;
        chroma_frame[i] = 128;
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
    result_output << input_file.filename() << ","
                  << input_file.width() << ","
                  << input_file.height() << ","
                  << type << ","
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

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            break;

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

Result run_trvl(InputFile &input_file)
{
    int frame_size = input_file.width() * input_file.height();
    int depth_buffer_size = frame_size * sizeof(short);
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

    while (!input_file.input_stream().eof())
    {
        input_file.input_stream().read(reinterpret_cast<char *>(depth_buffer.data()), depth_buffer_size);
        Timer compression_timer;
        // Update the TRVL pixel values with the raw depth pixels.
        for (int i = 0; i < frame_size; ++i)
        {
            trvl::update_pixel(trvl_pixels[i], depth_buffer[i], CHANGE_THRESHOLD, INVALIDATION_THRESHOLD);
        }

        // For the first frame, since there is no previous frame to diff, run vanilla RVL.
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
                pixel_diffs[i] = value - prev_pixel_values[i];
                prev_pixel_values[i] = value;
            }
            // Compress and decompress the difference.
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
        }

        auto depth_mat = create_depth_mat(input_file.width(), input_file.height(), depth_image.data());

        cv::imshow("Depth", depth_mat);
        if (cv::waitKey(1) >= 0)
            break;

        compressed_size_sum += rvl_frame.size();
        // The first frame goes through vanilla RVL which is lossless.
        float mse_value = mse(depth_buffer, depth_image);
        if (mse_value != 0.0f)
        {
            psnr_sum += 20.0f * log10(max(depth_buffer) / sqrt(mse_value));
        }
        else
        {
            ++zero_psnr_frame_count;
        }
        ++frame_count;
    }
    input_file.input_stream().close();

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
    // set parameters for scripting
    if (argc == 3)
    {
        rootPath = argv[1];
        std::cout << rootPath << " ";
        inFile = argv[2];
        std::cout << inFile << std::endl;
    }
    else if (argc == 7)
    {
        rootPath = argv[1];
        std::cout << rootPath << std::endl;
        inFile = argv[2];
        std::cout << inFile << " ";
        CHANGE_THRESHOLD = std::stoi(argv[3]);
        std::cout << CHANGE_THRESHOLD << " ";
        INVALIDATION_THRESHOLD = std::stoi(argv[4]);
        std::cout << INVALIDATION_THRESHOLD << " ";
        TRVL = std::stoi(argv[5]);
        if (TRVL == 1)
        {
            std::cout << " trvl ";
        }
        else
        {
            std::cout << " rvl ";
        }
        APPEND = std::stoi(argv[6]);
        std::cout << APPEND << std::endl;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " rootPath binName changeThreshold invalidationThreshold (0: rvl | 1:trvl) (0: overwrite | 1: append)" << std::endl;
        return 0;
    }

    const std::string RESULT_OUTPUT_FILE_PATH = "/home/sc/streamingPipeline/analysisData/temporal-rvl-data/result.csv";

    // char outPath[1024 * 2] = {0};
    // char prefix[1024] = {0};
    // strncpy(prefix, rootPath, strlen(rootPath) - 4);
    // sprintf(outPath, "%strvl/%s.png", prefix, inFile);
    // std::cout << outPath << std::endl;
    InputFile input_file(create_input_file(rootPath, inFile));

    if (APPEND == 1)
    {
        std::ofstream result_output(RESULT_OUTPUT_FILE_PATH, std::ios::app);
        if (TRVL == 1)
        {
            // reset_input_file(input_file);
            Result trvl_result(run_trvl(input_file));
            write_result_output_line(result_output, input_file, "trvl", CHANGE_THRESHOLD, INVALIDATION_THRESHOLD, trvl_result);
            result_output.close();
        }
        else
        {
            Result rvl_result(run_rvl(input_file));
            write_result_output_line(result_output, input_file, "rvl", 0, 0, rvl_result);
            result_output.close();
        }
    }
    else
    {
        std::ofstream result_output(RESULT_OUTPUT_FILE_PATH, std::ios::out);
        result_output << "filename,width,height,type,change_threshold,invalidation_threshold,";
        result_output << "average_compression_time,average_decompression_time,";
        result_output << "compression_ratio,average_psnr" << std::endl;

        if (TRVL == 1)
        {
            // reset_input_file(input_file);
            Result trvl_result(run_trvl(input_file));
            write_result_output_line(result_output, input_file, "trvl", CHANGE_THRESHOLD, INVALIDATION_THRESHOLD, trvl_result);
            result_output.close();
        }
        else
        {
            Result rvl_result(run_rvl(input_file));
            write_result_output_line(result_output, input_file, "rvl", 0, 0, rvl_result);
            result_output.close();
        }
    }

    return 0;
}