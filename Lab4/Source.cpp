#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>  

using namespace std;
using namespace cv;

// Function to load an image from the disk
int* inputImage(int* width, int* height, const string& imagePath) {
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);  // Read the image as grayscale
    if (image.empty()) {
        cerr << "Error: Image not found." << endl;
        exit(EXIT_FAILURE);
    }

    *width = image.cols;  // Set the image width
    *height = image.rows;  // Set the image height

    int* imageData = new int[(*width) * (*height)];  // Dynamic allocation for image data
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            imageData[i * (*width) + j] = (int)image.at<uchar>(i, j);  // Store pixel values
        }
    }
    return imageData;  // Return the array containing pixel values
}

// Function to save the processed image to disk
void createImage(int* outputData, int width, int height, const string& outputPath) {
    Mat outputImage(height, width, CV_8U);  // Create an empty image
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            outputImage.at<uchar>(i, j) = (uchar)outputData[i * width + j];  // Set pixel values
        }
    }
    imwrite(outputPath, outputImage);  // Write the image to file
}

// Function to apply a low-pass filter to the image
void applyLowPassFilter(int* input, int* output, int width, int height, int kernelSize) {
    int offset = kernelSize / 2;
    int sumKernel = kernelSize * kernelSize;  // Total number of elements in the kernel

    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            int sum = 0;
            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    sum += input[(y + ky) * width + (x + kx)];  // Sum the values within the kernel
                }
            }
            output[y * width + x] = sum / sumKernel;  // Compute the average and set it to the output
        }
    }

    // Handle image borders by replicating the adjacent values
    for (int i = 0; i < width; i++) {
        output[i] = input[i];  // Top border
        output[(height - 1) * width + i] = input[(height - 1) * width + i];  // Bottom border
    }
    for (int i = 0; i < height; i++) {
        output[i * width] = input[i * width];  // Left border
        output[i * width + (width - 1)] = input[i * width + (width - 1)];  // Right border
    }
}

// Parallel version of the low-pass filter function using OpenMP
void applyLowPassFilterParallel(int* input, int* output, int width, int height, int kernelSize) {
    int offset = kernelSize / 2;
    int sumKernel = kernelSize * kernelSize;

    // Parallel processing of the image pixels
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

    // Handle the image borders in parallel
#pragma omp parallel for
    for (int i = 0; i < width; i++) {
        output[i] = input[i];  // Top border
        output[(height - 1) * width + i] = input[(height - 1) * width + i];  // Bottom border
    }

#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        output[i * width] = input[i * width];  // Left border
        output[i * width + (width - 1)] = input[i * width + (width - 1)];  // Right border
    }
}

int main() {
    int ImageWidth = 0, ImageHeight = 0;

    string imagePath = "../lena.png";
    int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

    int* outputData = new int[ImageWidth * ImageHeight];
    int kernelSize = 3;  // Kernel size for the filter

    clock_t start_s = clock();
    applyLowPassFilterParallel(imageData, outputData, ImageWidth, ImageHeight, kernelSize);
    clock_t stop_s = clock();

    string outputPath = "../filterlena2.png";
    createImage(outputData, ImageWidth, ImageHeight, outputPath);

    double TotalTime = (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
    cout << "Time with OpenMP: " << TotalTime << " ms" << endl;

    delete[] imageData;
    delete[] outputData;

    return 0;
}
// testing git
