#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>  // Include OpenMP header

using namespace std;
using namespace cv;

int* inputImage(int* width, int* height, const std::string& imagePath) {
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image" << endl;
        exit(EXIT_FAILURE);
    }

    *width = image.cols;
    *height = image.rows;

    int* imageData = new int[(*width) * (*height)];
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            imageData[i * (*width) + j] = (int)image.at<uchar>(i, j);
        }
    }
    return imageData;
}

void createImage(int* outputData, int width, int height, const string& outputPath) {
    Mat outputImage(height, width, CV_8U);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            outputImage.at<uchar>(i, j) = (uchar)outputData[i * width + j];
        }
    }
    imwrite(outputPath, outputImage);
}

void applyLowPassFilter(int* input, int* output, int width, int height, int kernelSize) {
    int offset = kernelSize / 2;
    int sumKernel = kernelSize * kernelSize;

    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            int sum = 0;
            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    sum += input[(y + ky) * width + (x + kx)];
                }
            }
            output[y * width + x] = sum / sumKernel;
        }
    }

    // Handle borders by copying the adjacent values (border replication)
    for (int i = 0; i < width; i++) {
        output[i] = input[i]; // Top border
        output[(height - 1) * width + i] = input[(height - 1) * width + i]; // Bottom border
    }
    for (int i = 0; i < height; i++) {
        output[i * width] = input[i * width]; // Left border
        output[i * width + (width - 1)] = input[i * width + (width - 1)]; // Right border
    }
}

void applyLowPassFilterParallel(int* input, int* output, int width, int height, int kernelSize) {
    int offset = kernelSize / 2;
    int sumKernel = kernelSize * kernelSize;

#pragma omp parallel for collapse(2)
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            int sum = 0;
            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    sum += input[(y + ky) * width + (x + kx)];
                }
            }
            output[y * width + x] = sum / sumKernel;
        }
    }

    // Handle borders
#pragma omp parallel for
    for (int i = 0; i < width; i++) {
        output[i] = input[i]; // Top border
        output[(height - 1) * width + i] = input[(height - 1) * width + i]; // Bottom border
    }

#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        output[i * width] = input[i * width]; // Left border
        output[i * width + (width - 1)] = input[i * width + (width - 1)]; // Right border
    }
}

int main() {
    int ImageWidth = 0, ImageHeight = 0;

    std::string imagePath = "C:\\Users\\DELL\\Desktop\\HPC Labs\\Project\\Lab4\\lena.png";
    int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

    int* outputData = new int[ImageWidth * ImageHeight];
    int kernelSize = 3; // Can be modified

    clock_t start_s = clock();
    applyLowPassFilterParallel(imageData, outputData, ImageWidth, ImageHeight, kernelSize);
    clock_t stop_s = clock();

    string outputPath = "C:\\Users\\DELL\\Desktop\\HPC Labs\\Project\\Lab4\\filterlena2.png";
    createImage(outputData, ImageWidth, ImageHeight, outputPath);

    double TotalTime = (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
    cout << "Time with OpenMP: " << TotalTime << " ms" << endl;

    delete[] imageData;
    delete[] outputData;

    return 0;
}
