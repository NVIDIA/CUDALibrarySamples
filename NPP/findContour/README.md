# Find Contour using NPP

## Description

This code demonstrates Finding contour of a image  using NPP library.

## Key Concepts

Image segmentation  

## Supported SM Architectures

  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)


# Architecture
- Find Contour Sample.

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- Install the [CUDA 11.4 toolkit and above](https://developer.nvidia.com/cuda-downloads).

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
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

NPP Library Version 11.4.0
CUDA Driver Version: 11.2
CUDA Runtime Version: 11.4

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

