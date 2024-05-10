#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>  
#include <mpi.h>

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




void applyLowPassFilterParallelOMP(int* input, int* output, int width, int height, int kernelSize) {
    int offset = kernelSize / 2;
    int sumKernel = kernelSize * kernelSize;
    int sum;
    // Parallelize the main computation loop
#pragma omp parallel for collapse(2) shared(input, output, width, height, offset, sumKernel) private(sum) schedule(static)
    for (int y = offset; y < height - offset; y++) {
        for (int x = offset; x < width - offset; x++) {
            sum = 0;
            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    int index = (y + ky) * width + (x + kx);
                    if (index >= 0 && index < width * height) {  // Check bounds
                        sum += input[index];  // Sum the values within the kernel
                    }
                }
            }
            output[y * width + x] = sum / sumKernel;  // Compute the average and set it to the output
        }
    }

    // Handle image borders without parallel sections
    // Top and Bottom borders
#pragma omp parallel for shared(input, output, width, height) schedule(static)
    for (int i = 0; i < width; i++) {
        output[i] = input[i];  // Top border
        output[(height - 1) * width + i] = input[(height - 1) * width + i];  // Bottom border
    }

    // Left and Right borders
#pragma omp parallel for shared(input, output, width, height) schedule(static)
    for (int i = 0; i < height; i++) {
        output[i * width] = input[i * width];  // Left border
        output[i * width + (width - 1)] = input[i * width + (width - 1)];  // Right border
    }
}



void applyLowPassFilterMPI(int* input, int* output, int width, int height, int kernelSize) {
    int offset = kernelSize / 2;
    int sumKernel = kernelSize * kernelSize;  // Total number of elements in the kernel

    // We need to consider the ghost rows that might be necessary for the calculation.
    // Assuming input includes enough rows above and below the actual data (if necessary).
    // The first and last process might not have previous or next neighbors, handle these cases separately.

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

    // Handling the image borders
    int start_row = offset;  // Start processing from the first full context row
    int end_row = height - offset;  // End at the last full context row

    // Top border (only if the current chunk includes the top image border)
    if (start_row == offset) {
        for (int i = 0; i < width; i++) {
            output[i] = input[i];  // Replicate the top row
        }
    }

    // Bottom border (only if the current chunk includes the bottom image border)
    if (end_row == height - offset) {
        for (int i = 0; i < width; i++) {
            output[(height - 1) * width + i] = input[(height - 1) * width + i];  // Replicate the bottom row
        }
    }

    // Side borders for all rows handled in the subset
    for (int i = 0; i < height; i++) {
        output[i * width] = input[i * width];  // Left border
        output[i * width + (width - 1)] = input[i * width + (width - 1)];  // Right border
    }
}

int main() {
    int ImageWidth = 0, ImageHeight = 0;

    string imagePath = "../lena.png";
    string outputPath = "../filterlena2.png";
    int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

    int* outputData = new int[ImageWidth * ImageHeight];
    int kernelSize = 5;  // Kernel size for the filter

    clock_t startTime = clock();

    applyLowPassFilter(imageData, outputData, ImageWidth, ImageHeight, kernelSize);
    createImage(outputData, ImageWidth, ImageHeight, outputPath);
    double elapsedTime = static_cast<double>(clock() - startTime) / CLOCKS_PER_SEC *1000;

    cout << "Total execution time with Seq: " << elapsedTime << " ms" << endl;


    clock_t startTime2 = clock();
    applyLowPassFilterParallelOMP(imageData, outputData, ImageWidth, ImageHeight,kernelSize);

    double elapsedTime2 = static_cast<double>(clock() - startTime) / CLOCKS_PER_SEC *1000;
    cout << "Total execution time with OMP: " << elapsedTime2 << " ms" << endl;


    delete[] imageData;
    delete[] outputData;



    return 0;
}


/*
int main()
{
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    int ImageWidth = 0, ImageHeight = 0;
    string imagePath = "lena.png";
    int* imageData = nullptr;
    int* localImageData = nullptr;
    int* outputData = nullptr;

    if (rank == 0) {
        imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
        outputData = new int[ImageWidth * ImageHeight];
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(&ImageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&ImageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = ImageHeight / size;
    int extra_rows = ImageHeight % size;

    int local_start_row = rank * rows_per_proc + min(rank, extra_rows);

    int local_end_row = local_start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);

    int local_height = local_end_row - local_start_row;

    localImageData = new int[ImageWidth * local_height];

    // Scatter the image data
    MPI_Scatter(imageData, local_height * ImageWidth, MPI_INT, localImageData, local_height * ImageWidth, MPI_INT, 0, MPI_COMM_WORLD);

    int* localOutputData = new int[ImageWidth * local_height];
    applyLowPassFilterMPI(localImageData, localOutputData, ImageWidth, local_height, 3);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before gathering the results


    MPI_Gather(localOutputData, local_height * ImageWidth, MPI_INT, outputData, local_height * ImageWidth, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        string outputPath = "filterlena2.png";
        createImage(outputData, ImageWidth, ImageHeight, outputPath);
        delete[] imageData;
        delete[] outputData;
        cout << "Total execution time with MPI:: " << (MPI_Wtime() - startTime) * 1000 << " ms" << endl;
    }

    delete[] localImageData;
    delete[] localOutputData;

    MPI_Finalize();

    return 0;

}
*/
