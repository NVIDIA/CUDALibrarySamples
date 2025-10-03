/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_MSC_VER)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "getopt.h"
#pragma warning(disable : 4819)
const std::string separator = "\\";
#else
#include <getopt.h>
const std::string separator = "/";
#endif
#include <fstream>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <nvtiff.h>

using perfclock = std::chrono::high_resolution_clock;

#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))

#define CHECK_CUDA(call)                                                                                                \
    {                                                                                                                   \
        cudaError_t err = call;                                                                                         \
        if (cudaSuccess != err) {                                                                                       \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }

#define CHECK_NVTIFF(call)                                                                             \
    {                                                                                                  \
        nvtiffStatus_t _e = (call);                                                                    \
        if (_e != NVTIFF_STATUS_SUCCESS) {                                                             \
            fprintf(stderr, "nvtiff error code %d in file '%s' in line %i\n", _e, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

//#define LIBTIFF_TEST
#ifdef LIBTIFF_TEST
#include <tiffio.h>
#endif

#define MAX_STR_LEN	(256)
void write_pnm(const char* filename, unsigned char* chan, uint32_t ld, uint32_t width, uint32_t height, uint32_t BPP, uint32_t numcomp,
    uint32_t write_out_numcomp)
{
    std::ofstream rOutputStream(filename);
    if (!rOutputStream) {
        std::cerr << "Cannot open output file: " << filename << std::endl;
        return;
    }
    if (numcomp == 1) {
        rOutputStream << "P5\n";
    } else {
        rOutputStream << "P6\n";
    }

    rOutputStream << "#nvTIFF\n";
    rOutputStream << width << " " << height << "\n";
    rOutputStream << (1 << BPP) - 1 << "\n";
    if (BPP == 8) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < write_out_numcomp; c++) {
                    rOutputStream << chan[(y * ld + x) * numcomp + c];
                }
            }
        }
    } else {
        uint16_t* chan16 = reinterpret_cast<uint16_t*>(chan);
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++) {
                for (uint32_t c = 0; c < write_out_numcomp; c++) {
                    uint32_t pix_val = chan16[(y * ld + x) * numcomp + c];
                    rOutputStream << static_cast<unsigned char>((pix_val) >> 8) << static_cast<unsigned char>((pix_val)&0xff);
                }
            }
        }
    }
    return;
}

int write_image(std::string input_filename, nvtiffImageInfo& image_info, unsigned char* chan, uint32_t image_id)
{
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = input_filename.rfind(separator);
    std::string out_filename =
        (std::string::npos == position) ? input_filename : input_filename.substr(position + 1, input_filename.size());
    position = out_filename.rfind(".");
    out_filename = (std::string::npos == position) ? out_filename : out_filename.substr(0, position);
    out_filename += "_nvtiff_out_" + std::to_string(image_id);
    uint32_t num_samples = image_info.samples_per_pixel;
    uint32_t samples_written_to_file = image_info.samples_per_pixel;
    if (image_info.samples_per_pixel == 3 || image_info.samples_per_pixel == 4 ||
        image_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        out_filename += ".ppm";
        samples_written_to_file = 3;
    } else if (image_info.samples_per_pixel == 1) {
        out_filename += ".pgm";
    } else {
        printf("Unable to write image with %d samples per pixel, continuing to the next image..\n", image_info.samples_per_pixel);
        return EXIT_SUCCESS;
    }
    uint32_t bits_per_sample = image_info.bits_per_sample[0];
    if (image_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
        num_samples = 3;
        samples_written_to_file = 3;
        bits_per_sample = 16;
    }

    if (bits_per_sample == 16 || bits_per_sample == 8) {
        write_pnm(out_filename.c_str(), chan, image_info.image_width, image_info.image_width, image_info.image_height, bits_per_sample,
            num_samples, samples_written_to_file);
    } else {
        printf("Unable to write to file for this set of tiff image, continuing to next image\n");
    }

    return EXIT_SUCCESS;
}

static void usage(const char *pname) {
	
	fprintf(stdout, 
		"Usage:\n"
		"%s [options] -f|--file <TIFF_FILE>\n"
		"\n"
		"General options:\n"
		"\n"
		"\t-d DEVICE_ID\n"
		"\t--device DEVICE_ID\n"
		"\t\tSpecifies the GPU to use for images decoding/encoding.\n"
		"\t\tDefault: device 0 is used.\n"
		"\n"
		"\t-v\n"
		"\t--verbose\n"
		"\t\tPrints some information about the decoded TIFF file.\n"
		"\n"
		"\t-h\n"
		"\t--help\n"
		"\t\tPrints this help\n"
		"\n"
		"Decoding options:\n"
		"\n"
                "\t-f TIFF_FILE\n"
		"\t--file TIFF_FILE\n"
		"\t\tSpecifies the TIFF file to decode. The code supports both single and multi-image\n"
		"\t\ttiff files with the following limitations:                                      \n"
                "\t\t  * color space must be either Grayscale (PhotometricInterp.=1) or RGB (=2)     \n"
                "\t\t  * image data compressed with LZW (Compression=5) or uncompressed              \n"
                "\t\t  * pixel components stored in \"chunky\" format (RGB..., PlanarConfiguration=1)\n"
		"\t\t    for RGB images                                                              \n"
                "\t\t  * image data must be organized in Strips, not Tiles                           \n"
                "\t\t  * pixels of RGB images must be represented with at most 4 components          \n"
                "\t\t  * each component must be represented exactly with:                            \n"
                "\t\t      * 8 bits for LZW compressed images                                        \n"
                "\t\t      * 8, 16 or 32 bits for uncompressed images                                \n"
                "\t\t  * all images in the file must have the same properties                        \n"
		"\n"
		"\t-b BEG_FRM\n"
		"\t--frame-beg BEG_FRM\n"
		"\t\tSpecifies the image id in the input TIFF file to start decoding from.  The image\n"
		"\t\tid must be a value between 0 and the total number of images in the file minus 1.\n"
		"\t\tValues less than 0 are clamped to 0.\n"
		"\t\tDefault: 0\n"
		"\n"
		"\t-e END_FRM\n"
		"\t--frame-end END_FRM\n"
		"\t\tSpecifies the image id in the input TIFF file to stop  decoding  at  (included).\n"
		"\t\tThe image id must be a value between 0 and the total number  of  images  in  the\n"
		"\t\tfile minus 1.  Values greater than num_images-1  are  clamped  to  num_images-1.\n"
		"\t\tDefault:  num_images-1.\n"
		"\n"
		"\t-m\n"
		"\t--memtype TYPE\n"
		"\t\tSpecifies the type of memory used to hold  the  TIFF  file  content:  pinned  or\n"
		"\t\tpageable.  Pinned memory is used if 'p' is specified. Pageable memory is used if\n"
		"\t\t'r' is specified.  In case of pinned memory,  file  content  is  not  copied  to\n"
		"\t\tdevice memory before the decoding process (with a resulting performance  impact)\n"
		"\t\tunless the option -c is also specified (see below).\n"
		"\t\tDefualt: r (pageable)\n"
		"\n"
		"\t-c\n"
		"\t--copyh2d\n"
		"\t\tSpecifies to copy the file data to device memory in case the -m option specifies\n"
		"\t\tto use pinned memory.  In case of pageable memory this  option  has  no  effect.\n"
		"\t\tDefault: off.\n"
		"\n"
		"\t--decode-out NUM_OUT\n"
		"\t\tEnables the writing of selected images from the decoded  input  TIFF  file  into\n"
		"\t\tseparate BMP files for inspection.  If no argument is  passed,  only  the  first\n"
		"\t\timage is written to disk,  otherwise  the  first  NUM_OUT  images  are  written.\n"
		"\t\tOutput files are named outImage_0.bmp, outImage_1.bmp...\n"
		"\t\tDefualt: disabled.\n"
		"\n"
		"Encoding options:\n"
		"\n"
		"\t-E\n"
		"\t--encode\n"
		"\t\tThis option enables the encoding of the raster images obtained by  decoding  the\n"
		"\t\tinput TIFF file.  The images are divided into strips, compressed  with  LZW and,\n"
		"\t\toptionally, written into an output TIFF file.\n"
		"\t\tDefault: disabled.\n"
		"\n"
		"\t--encode-out\n"
		"\t\tEnables the writing of the compressed  images  to  an  output  TIFF  file named\n"
		"\t\toutFile.tif.\n"
		"\t\tDefualt: disabled.\n",
		pname);

	exit(EXIT_FAILURE);
}

bool check_identical(nvtiffImageInfo_t * image_info, uint32_t num_images) {
	
	bool identical = true;
    // now check that all subfiles have the same properties
    for(unsigned int i = 1; i < num_images; i++) {
		if ((image_info[i].image_width     != image_info[i -1].image_width)            ||
            (image_info[i].image_height    != image_info[i -1].image_height)            ||
            (image_info[i].samples_per_pixel != image_info[i -1].samples_per_pixel) ||
			(image_info[i].bits_per_pixel  != image_info[i -1].bits_per_pixel) ||
            memcmp(image_info[i].sample_format,
               image_info[i-1].sample_format,
               sizeof(short)*image_info[i].samples_per_pixel)||
            memcmp(image_info[i].bits_per_sample,
               image_info[i-1].bits_per_sample,
               sizeof(short)*image_info[i].samples_per_pixel)) {
				identical = false;
				break;
           }
    }
	return identical;
}

int main(int argc, char **argv) {

	int devId = 0;

	char *fname = NULL;

	int verbose = 0;
	int decWriteOutN = 0;

	int frameBeg = INT_MIN;
	int frameEnd = INT_MAX;
	int decodeRange = 0;
	int doEncode = 0;
	int encWriteOut = 0;

	int och;
	while(1) {
		int option_index = 0;
		static struct option long_options[] = {
			{      "file", required_argument, 0, 'f'},
			{    "device", required_argument, 0, 'd'},
			{"decode-out", optional_argument, 0,   1},
			{ "frame-beg", required_argument, 0, 'b'},
			{ "frame-end", required_argument, 0, 'e'},
			{   "memtype", required_argument, 0, 'm'},
			{   "copyh2d", required_argument, 0, 'c'},
			{   "verbose",       no_argument, 0, 'v'},
			{    "encode",       no_argument, 0, 'E'},
			{"rowsxstrip", required_argument, 0, 'r'},
			{"stripalloc", required_argument, 0, 's'},
			{"encode-out", optional_argument, 0,   2},
			{      "help",       no_argument, 0, 'h'},
			{           0,                 0, 0,   0}
		};

		och = getopt_long(argc, argv, "f:d:vo::hb:e:m:cEr:s:", long_options, &option_index);
		if (och == -1) break;
		switch (och) {
			case   0:// handles long opts with non-NULL flag field
				break;
			case 'd':
				devId = atoi(optarg);
				break;
			case 'f':
				fname = strdup(optarg);
				break;
			case 'b':
				frameBeg = atoi(optarg);
				decodeRange = 1;
				break;
			case 'e':
				frameEnd = atoi(optarg);
				decodeRange = 1;
				break;
			case 'v':
				verbose++;
				break;
			case   1:
				decWriteOutN = 1;
				if(!optarg                 &&
				   argv[optind]    != NULL &&
				   argv[optind][0] != '-') {

					decWriteOutN = atoi(argv[optind++]);
				}
				break;
			case 'E':
				doEncode = 1;
				break;
			case   2:
				encWriteOut = 1;
				break;
			case 'h':
			case '?':
				usage(argv[0]);
			default:
				fprintf(stderr, "unknown option: %c\n", och);
				usage(argv[0]);
		}
	}

	if (!fname) {
		fprintf(stderr, "Please specify a TIFF file with the -f option!\n");
		usage(argv[0]);
	}

	if (frameBeg > frameEnd) {
		fprintf(stderr, "Invalid frame range!\n");
		usage(argv[0]);
	}

	CHECK_CUDA(cudaSetDevice(devId));

	cudaDeviceProp props;

	printf("\nUsing GPU:\n");
	CHECK_CUDA(cudaGetDeviceProperties(&props, devId));
	printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
			devId, props.name, props.multiProcessorCount,
			props.maxThreadsPerMultiProcessor,
			props.major, props.minor,
			props.ECCEnabled?"on":"off");
	printf("\n");

	// dummy allocation to initialize subsystems
	unsigned char *dummy;
	CHECK_CUDA(cudaMalloc(&dummy, 1024*1024*10));
	CHECK_CUDA(cudaFree(dummy));

	cudaStream_t stream;
	CHECK_CUDA(cudaStreamCreate(&stream));


	nvtiffStream_t tiff_stream;
	nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
	CHECK_NVTIFF(nvtiffDecoderCreate(&decoder,
        nullptr, nullptr, 0));

    CHECK_NVTIFF(nvtiffStreamParseFromFile(fname, tiff_stream));

	uint32_t num_images = 0;
    CHECK_NVTIFF(nvtiffStreamGetNumImages(tiff_stream, &num_images));
	std::vector<nvtiffImageInfo_t> image_info(num_images);
    std::vector<uint8_t*> nvtiff_out(num_images);
    std::vector<size_t> nvtiff_out_size(num_images);
    
	// BEGIN work (possibly) overlapped with H2D copy of the file data
	if (verbose) {
		CHECK_NVTIFF(nvtiffStreamPrint(tiff_stream));
	}
	
	frameBeg = fmax(frameBeg, 0);
	frameEnd = fmin(frameEnd, num_images-1);
	const int nDecode = frameEnd-frameBeg+1;

	for (uint32_t image_id = 0; image_id < num_images; image_id++) {
        CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, image_id, &image_info[image_id]));
        nvtiff_out_size[image_id] = DIV_UP((size_t)image_info[image_id].bits_per_pixel * image_info[image_id].image_width, 8) *
                                    (size_t)image_info[image_id].image_height;
        if (image_info[image_id].photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
            nvtiff_out_size[image_id] = image_info[image_id].image_width * image_info[image_id].image_height * 3 * sizeof(uint16_t);
        }
        CHECK_CUDA(cudaMalloc(&nvtiff_out[image_id], nvtiff_out_size[image_id]));
    }

	printf("Decoding %u, images [%d, %d], from file %s... ",
		nDecode,
		frameBeg,
		frameEnd,
		fname);
	fflush(stdout);


	auto decode_start = perfclock::now();
	if (!decodeRange) {
		CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, nvtiff_out.data(), stream));
	} else { 
		CHECK_NVTIFF(nvtiffDecodeRange(tiff_stream, decoder, frameBeg, nDecode, nvtiff_out.data(), stream));
	}
	CHECK_CUDA(cudaStreamSynchronize(stream));
	auto decode_end = perfclock::now();
    double decode_time = std::chrono::duration<float>(decode_end - decode_start).count();

	printf("done in %lf secs\n\n", decode_time);

	if (decWriteOutN) {
        const uint32_t nout = std::min(decWriteOutN, nDecode);

        printf("Writing images for the first %d subfile(s)...\n", nout);
        fflush(stdout);

        for (uint32_t image_id = 0; image_id < nout; image_id++) {
            auto& info = image_info[image_id];
            std::vector<uint8_t> imageOut_h(nvtiff_out_size[image_id]);
            CHECK_CUDA(cudaMemcpy(imageOut_h.data(), nvtiff_out[image_id], nvtiff_out_size[image_id], cudaMemcpyDeviceToHost));
            write_image(fname, info, imageOut_h.data(), image_id);
        }
    }

#ifdef LIBTIFF_TEST
	TIFF* tif = TIFFOpen(fname, "r");
	if (tif) {

		// we alredy know that all subfiles have the same porperties
		uint32_t *raster;
		raster = (uint32_t *)_TIFFmalloc(tiffData->subFiles[0].ncol*tiffData->subFiles[0].nrow * sizeof (uint32_t));

		printf("\tDecoding with libTIFF... "); fflush(stdout);
		auto decode_start = perfclock::now();
		for(int i = 0; i < tiffData->nSubFiles; i++) {
			if (!TIFFReadRGBAImage(tif,
					       tiffData->subFiles[i].ncol,
					       tiffData->subFiles[i].nrow,
					       raster, 0)) {
				fprintf(stderr, "Error while decoding image %d with libTiff\n", i);
				break;
			}
			TIFFReadDirectory(tif);
		}
		auto decode_end = perfclock::now();
		double decode_time = std::chrono::duration<float>(decode_end - decode_start).count();
		printf("done in %lf secs\n\n", decode_time);

		_TIFFfree(raster);
		TIFFClose(tif);
	}
#endif
	bool identical_multi_tiff = check_identical(image_info.data(), num_images);
	if(!identical_multi_tiff && doEncode){
		printf("Encoding will be skipped since the images within the tiff file do not have identical properties...\n");
	}
	// TODO check identical
	if (doEncode && identical_multi_tiff) {

		unsigned int nrow              = image_info[0].image_height;
		unsigned int ncol              = image_info[0].image_width;
		unsigned int photometricInt    = (unsigned int)image_info[0].photometric_int;
		unsigned int planarConf        = (unsigned int)image_info[0].planar_config;
		unsigned short pixelSize       = image_info[0].bits_per_pixel/8;
		unsigned short samplesPerPixel = image_info[0].samples_per_pixel;
		unsigned short sampleFormat    = image_info[0].sample_format[0];

		unsigned short *bitsPerSample = (unsigned short *)malloc(sizeof(*bitsPerSample)*samplesPerPixel);
		memcpy(bitsPerSample,
			image_info[0].bits_per_sample,
		       sizeof(*bitsPerSample)*samplesPerPixel);

		CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
        CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
		tiff_stream = NULL;
		decoder = NULL;

		unsigned int nSubFiles = nDecode;

		printf("Encoding %u, %s %ux%u images ... ",
			nDecode, image_info[0].photometric_int == 2 ? "RGB" : "Grayscale",
			image_info[0].image_width, image_info[0].image_height);
		fflush(stdout);

        auto enc_start = perfclock::now();

		nvtiffEncoder_t encoder;
		CHECK_NVTIFF(nvtiffEncoderCreate(&encoder, nullptr, nullptr, stream));
		nvtiffEncodeParams_t params;
		CHECK_NVTIFF(nvtiffEncodeParamsCreate(&params));
		CHECK_NVTIFF(nvtiffEncodeParamsSetImageInfo(params, &image_info[0]));
		CHECK_NVTIFF(nvtiffEncodeParamsSetInputs(params, nvtiff_out.data(), nSubFiles));
		CHECK_NVTIFF(nvtiffEncode(encoder, &params, 1, stream));
		CHECK_NVTIFF(nvtiffEncodeFinalize(encoder, &params, 1, stream));

		CHECK_CUDA(cudaStreamSynchronize(stream));
		auto enc_end = perfclock::now();
		double enc_time = std::chrono::duration<float>(enc_end - enc_start).count();

	    size_t stripSizeTotal = 0;
        size_t metadataSize = 0;
		CHECK_NVTIFF(nvtiffGetBitstreamSize(encoder, &params, 1, &metadataSize, &stripSizeTotal));
		printf("done in %lf secs (compr. ratio: %.2lfx)\n\n",
				enc_time, double(nvtiff_out_size[0])*nSubFiles/stripSizeTotal);

		if (encWriteOut) {
			printf("\tWriting %u compressed images to TIFF file... ", nDecode); fflush(stdout);
			auto write_start = perfclock::now();
			CHECK_NVTIFF(nvtiffWriteTiffFile(encoder, &params, 1, "outFile.tif", stream));
			auto write_end = perfclock::now();
			double write_time = std::chrono::duration<float>(write_end - write_start).count();
			printf("done in %lf secs\n\n", write_time);

		}
        CHECK_NVTIFF(nvtiffEncodeParamsDestroy(params, stream));
		CHECK_NVTIFF(nvtiffEncoderDestroy(encoder, stream));

#ifdef LIBTIFF_TEST
		tif = TIFFOpen("libTiffOut.tif", "w");
		if (tif) {

			unsigned char **imageOut_h = (unsigned char **)Malloc(sizeof(*imageOut_h)*nDecode);
			for(unsigned int i = 0; i < nDecode; i++) {
				imageOut_h[i] = (unsigned char *)Malloc(sizeof(*imageOut_h)*imageSize);
				CHECK_CUDA(cudaMemcpy(imageOut_h[i],
							imageOut_d[i],
							imageSize,
							cudaMemcpyDeviceToHost));
			}

			size_t stripSize = sizeof(**imageOut_h)*encRowsPerStrip*ncol*pixelSize;

			printf("\tEncoding with libTIFF... "); fflush(stdout);
			__t = Wtime();
			for(unsigned int i = 0; i < nDecode; i++) {

				TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, nrow);
				TIFFSetField(tif, TIFFTAG_IMAGELENGTH, ncol);
				TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
				TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
				TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, photometricInt);
				TIFFSetField(tif, TIFFTAG_FILLORDER, 1);
				TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);
				TIFFSetField(tif, TIFFTAG_PLANARCONFIG, planarConf);
				TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, encRowsPerStrip);

				for(unsigned int j = 0; j < nStripOut; j++) {

					unsigned int currStripSize = stripSize;
					if (j == nStripOut-1) {
						currStripSize = imageSize - j*stripSize;
					}

					if (-1 == TIFFWriteEncodedStrip(tif,
								j,
								imageOut_h[i]+j*stripSize,
								currStripSize)) {

						fprintf(stderr, "Error while encoding image %d with libTiff\n", i);
						break;
					}
				}
				// need to find a way to have libTiff to encode in 
				// memory without writing to disk the last direnctory
				// after each TIFFWriteDirectory() call
				TIFFWriteDirectory(tif);
				//TIFFRewriteDirectory(tif);
			}
			__t = Wtime()-__t;
			printf("done in %lf secs\n\n", __t);

			TIFFClose(tif);
		}
#endif
	}

	// cleanup
	for(unsigned int i = 0; i < nDecode; i++) {
		CHECK_CUDA(cudaFree(nvtiff_out[i]));
	}

	free(fname);
	
	if(tiff_stream)	{
		CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
	}
    
	if(decoder){
	    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
	 }
	
	CHECK_CUDA(cudaStreamDestroy(stream));

	CHECK_CUDA(cudaDeviceReset());

	return 0;
}
