# Image Processing with Parallel Computing

This repository contains C++ code for performing image processing, specifically applying a low-pass filter (averaging filter) to images. The project utilizes different technologies including OpenCV for image handling, OpenMP for multithreading, and MPI (Message Passing Interface) for distributed computing.

## Prerequisites

Before you can run this project, you need to have the following installed:
- C++ compiler (e.g., GCC, Clang)
- [OpenCV](https://opencv.org/releases/) - Used for image manipulation.
- [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/) - Used for multi-threading.
- [MPI](https://www.mpich.org/downloads/) - Used for distributed computing.

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/MohamadAbdelmoniem/Low-Pass-Filter.git
cd Low-Pass-Filter
```

## Features

- **Input Image**: Load a grayscale image from the disk using OpenCV.
- **Low-Pass Filter**: Apply a simple averaging filter to smooth the image. This is done using three methods:
  - Sequential processing
  - Parallel processing with OpenMP
  - Distributed processing with MPI (commented out in the main function; can be enabled if running in a distributed environment)

- **Output Image**: Save the processed image back to the disk.

## Code Explanation

- **Image Input/Output**: The functions `inputImage` and `createImage` handle reading and writing images using OpenCV. `inputImage` reads an image in grayscale, converts it to a 1D array, and returns it along with its dimensions. `createImage` takes processed image data and saves it as a new image file.

- **Low-Pass Filter Implementation**:
  - **Sequential**: The `applyLowPassFilter` function applies the averaging filter by iterating over each pixel, calculating the mean of the surrounding pixels within the kernel's size, and setting the new pixel value.
  - **Parallel with OpenMP**: `applyLowPassFilterParallelOMP` enhances performance by using OpenMP to parallelize the filtering process. The computation over the image matrix is divided among multiple threads, significantly speeding up the operation.
  - **Distributed with MPI**: `applyLowPassFilterMPI` is designed for environments where MPI is configured. It distributes parts of the image to different processes, each applying the filter to its section of the image. This is useful for very large images or high-performance requirements.

- **Performance Measurement**: Execution time for the filtering process is measured and printed, providing insights into the efficiency gains from parallel processing.


- This project uses [OpenCV](https://opencv.org/) for handling image operations.
- Parallel processing components utilize [OpenMP](https://www.openmp.org/) and [MPI](https://www.mpich.org/).

---

This updated README is now linked correctly to your GitHub repository, making it easy for collaborators and users to find and use your project.
