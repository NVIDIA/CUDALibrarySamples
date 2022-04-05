# Python TIFF Image decoding Example using nvTIFF Library

## Description

Python script to use TIFF Image decoding using nvTIFF library.

## Key Concepts

Image decoding from NVTIFF Library

## Supported SM Architectures

  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

x86_64

## CUDA APIs involved

[NVTIFF](https://docs.nvidia.com/cuda/nvTIFF/index.html)

# Prerequisites
- A Linux system with recent NVIDIA drivers.
- Install the [CUDA 11.6 toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake (3.13 or later)
- Python 3.6
- nvTIFF package


## Python package needed
[cupy](https://docs.cupy.dev/en/stable/install.html) 
```
$  pip install cupy
$
```

[numpy](https://numpy.org/install/) 
```
$  pip install numpy
$
```

[tifffile](https://pypi.org/project/tifffile/) 
```
$  pip install tifffile
$
```

# Install nvTIFF from python wheel located in folder [nvTIFF Python Wheel](nvTIFF-Python-Whl/)
```
$  pip install nvtiff-0.1.0-cp36-cp36m-linux_x86_64.whl
$
```

```
Defaulting to user installation because normal site-packages is not writeable

Processing ./nvtiff-0.1.0-cp36-cp36m-linux_x86_64.whl

Requirement already satisfied: cupy in /usr/local/lib/python3.6/dist-packages (from nvtiff==0.1.0) (9.0.0)

Requirement already satisfied: numpy in /home/mahesh/.local/lib/python3.6/site-packages (from nvtiff==0.1.0) (1.19.5)

Requirement already satisfied: fastrlock>=0.5 in /home/mahesh/.local/lib/python3.6/site-packages (from cupy->nvtiff==0.1.0) (0.6)

Installing collected packages: nvtiff

Successfully installed nvtiff-0.1.0

```

# Set the nvTIFF library path 
```
$  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/libnvtiff/lib64/
$
```

# Usage

## Testing Python scripts
```
$ python3 nvtiff_test.py -h
```

```
usage: nvtiff_test.py [-h] [-o OUTPUT_FILE_PREFIX] [-s] [-c] [-p]
                      [-r SUBFILE_RANGE]
                      tiff_file


positional arguments:
  tiff_file             tiff file to decode.


optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE_PREFIX, --output_file_prefix OUTPUT_FILE_PREFIX
                        Output file prefix to save decoded data. Will save one
                        file per image in tiff file.
  -s, --return_single_array
                        Return single array from nvTiff instead of list of
                        arrays
  -c, --check_output    Compare nvTiff output to reference CPU result
  -p, --use_pinned_mem  Read TIFF data from pinned memory.
  -r SUBFILE_RANGE, --subfile_range SUBFILE_RANGE
                        comma separated list of starting and ending file
                        indices to decode, inclusive

```

Example:

nvTIFF decoding example output on GV100, Ubuntu 16.04, CUDA 11.6

```
$ python3 nvtiff_test.py bali_notiles.tif 

```

```
Command line arguments:

	tiff_file: bali_notiles.tif
	return_single_array: False
	output_file_prefix: None
	check_output: False
	use_pinned_mem: False
	subfile_range: None

Time for tifffile:

	decode:   0.010347366333007812 s
	h2d copy: 0.0010058879852294922 s
	total:    0.011353254318237305 s

Time for nvTiff:

	open: 0.002551555633544922 s
	decode: 0.0005545616149902344 s
	total:  0.0031061172485351562 s

```
