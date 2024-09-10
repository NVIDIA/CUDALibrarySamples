# Find Contour Detection using NPP+

## Description

This code demonstrates **Contour Detection** using the **Nvidia Performance Primitives (NPP+)** library. Contour detection involves identifying the boundaries of objects within an image, which is a key task in computer vision and image processing.

## Key Concepts

- **Image Contour Detection**: Detecting the edges and boundaries of objects within images to analyze shapes, objects, and regions of interest.

## Supported SM Architectures

This example is optimized for the following Nvidia GPU architectures:

- [SM 7.0](https://developer.nvidia.com/cuda-gpus)
- [SM 7.2](https://developer.nvidia.com/cuda-gpus)
- [SM 7.5](https://developer.nvidia.com/cuda-gpus)
- [SM 8.0](https://developer.nvidia.com/cuda-gpus)

## Supported Operating Systems

- Linux
- Windows

## Supported CPU Architecture

- x86_64

## Getting Started with NPP+

To begin using NPP+, download the latest version of the library and install it on your Nvidia GPU-equipped machine.

- [Download NPP+](https://developer.nvidia.com/nppplus-downloads)

### NPP+ APIs Involved
This example uses the [NPP+](https://docs.nvidia.com/cuda/nppplus/introduction.html) library, which provides optimized functions for image processing tasks on Nvidia GPUs.

## Architecture

This project includes the following main components:

- **Find Contour Sample**: Demonstrates how to efficiently detect contours in images using NPP+ functions.

## Prerequisites

Before building and running the example, ensure you have the following:

- A Linux or Windows system with a supported Nvidia GPU.
- The latest [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) installed.
- Install the [CUDA 12.0 toolkit and above](https://developer.nvidia.com/cuda-downloads).

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake .. -DNPP_PLUS_PATH=/usr/lib/x86_64-linux-gnu 
$ make
```

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open findContour.sln project in Visual Studio 15 2017 and build
```


# Usage
./findContour -h
```
Usage: ./findContour 
Parameters: 
	number-of-images	:	Use 2 number of images

```
Example:
```
```
$  ./findContour 

./findContour 

NPP Library Version 0.9.0
CUDA Driver  Version: 12.7
CUDA Runtime Version: 12.6

Input file load succeeded.
CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u succeeded, compressed label count is 249. ![Label count list](/NPP/findContour/contour_info.log)
nID 1 BB 190 0 539 299  
ID 1 Cnt 1 505 285  
ID 1 Cnt 2 539 275  
ID 1 Cnt 3 533 263  
ID 1 Cnt 4 537 269  
ID 1 Cnt 5 513 265  
ID 1 Cnt 6 512 265  
ID 1 Cnt 7 511 26




Input Image
Input Image - CircuitBoard_2048x1024_8u.raw - image name has image info such as width=2048, height=1024 and data type=8bit unsigned.
![CircuitBoard_2048x1024_8u.raw ](/NPP/findContour/CircuitBoard_2048x1024_8u.jpg)

Output images at different stages of Contour –
Figure 1 shows (CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.raw) the results from the nppiLabelMarkersUF_8u32u_C1R_Ctx()  API and which generate the 1 channel 8-bit to 32-bit unsigned integer label markers image.

Image info – 8 way connectivity width=2048, height=1024 and datatype=32bit unsigned

![CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.raw](/NPP/findContour/CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.jpg)

Figure 2 shows (CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.raw) the results from the nppiCompressMarkerLabelsUF_32u_C1IR() API and typical operation 1 channel 32-bit unsigned integer in place connected region marker label renumbering for output from nppiLabelMarkersUF functions only with numbering sparseness elimination.

Image info – 8 way connectivity width=2048, height=1024 and datatype=32bit unsigned
![CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.raw](/NPP/findContour/CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.jpg)


Figure 3 shows (CircuitBoard_Contours_8Way_2048x1024_8u.raw) the results from the 
nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx() API and typical operation 1 channel 32-bit unsigned integer connected region marker label renumbered from a previous call to nppiCompressMarkerLabelsUF or nppiCmpressMarkerLabelsUFBatch functions to eliminate 
label ID sparseness.

Image info – 8 way connectivity width=2048, height=1024 and datatype=8bit unsigned
![CircuitBoard_Contours_8Way_2048x1024_8u.raw](/NPP/findContour/CircuitBoard_Contours_8Way_2048x1024_8u.jpg)

Figure 4 shows (CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.raw) the results from the 
nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R() API and typical operation includes 1 channel connected region contours image to output contours geometry list in host memory. Note that ALL input and output data for the function MUST be in host memory. Also nFirstContourID and nLastContourID allow only a portion of the contour geometry lists in the image to be output. Note that the geometry list for each contour will begin at pContoursGeometryListsHost[pContoursPixelStartingOffsetHost[nContourID] * sizeof(NppiContourPixelGeometryInfo). Also note that the ordered contour geometry list is contained in the oContourOrderedGeometryLocation object within the NppiContourPixelGeometryInfo object for each contour pixel and that this location information is the only valid information in the object relevant to that ordered pixel 

Image info – 8 way connectivity width=2048, height=1024 and datatype=8bit unsigned
![CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.raw](/NPP/findContour/CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.jpg)



```

