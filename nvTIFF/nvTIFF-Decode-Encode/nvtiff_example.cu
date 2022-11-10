/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_MSC_VER)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#include <windows.h>
#include <chrono>
#include "getopt.h"
#  pragma warning(disable:4819)
#else
#include <getopt.h>
#endif
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include "nvTiff_utils.h"
#include "cudamacro.h"
#include <nvTiff.h>

#define CHECK_NVTIFF(call)                                                \
    {                                                                       \
        nvtiffStatus_t _e = (call);                                       \
        if (_e != NVTIFF_STATUS_SUCCESS)                                   \
        {                                                                   \
            std::cerr<< "nvTiff failure: '#" << _e<<std::endl;            \
            exit(EXIT_FAILURE);                                           \
        }                                                                    \
    }

//#define LIBTIFF_TEST
#ifdef LIBTIFF_TEST
#include <tiffio.h>
#endif

#define MAX_STR_LEN	(256)

// quick and dirty BMP file writer
static void writeBMPFile(const char *filename, unsigned char *chan, int LD, int WIDTH, int HEIGHT, int BPP, int IS_GREYSCALE) {

	unsigned int headers[13];
	FILE * outfile;
	int extrabytes;
	int paddedsize;
	int x; int y; int n;
	int red, green, blue;

	extrabytes = 4 - ((WIDTH * 3) % 4);                 // How many bytes of padding to add to each
	// horizontal line - the size of which must
	// be a multiple of 4 bytes.
	if (extrabytes == 4)
		extrabytes = 0;

	paddedsize = ((WIDTH * 3) + extrabytes) * HEIGHT;

	// Headers...
	// Note that the "BM" identifier in bytes 0 and 1 is NOT included in these "headers".

	headers[0]  = paddedsize + 54;      // bfSize (whole file size)
	headers[1]  = 0;                    // bfReserved (both)
	headers[2]  = 54;                   // bfOffbits
	headers[3]  = 40;                   // biSize
	headers[4]  = WIDTH;  // biWidth
	headers[5]  = HEIGHT; // biHeight

	// Would have biPlanes and biBitCount in position 6, but they're shorts.
	// It's easier to write them out separately (see below) than pretend
	// they're a single int, especially with endian issues...

	headers[7]  = 0;                    // biCompression
	headers[8]  = paddedsize;           // biSizeImage
	headers[9]  = 0;                    // biXPelsPerMeter
	headers[10] = 0;                    // biYPelsPerMeter
	headers[11] = 0;                    // biClrUsed
	headers[12] = 0;                    // biClrImportant

	outfile = Fopen(filename, "wb");

	//
	// Headers begin...
	// When printing ints and shorts, we write out 1 character at a time to avoid endian issues.
	//

	fprintf(outfile, "BM");

	for (n = 0; n <= 5; n++)
	{
		fprintf(outfile, "%c", headers[n] & 0x000000FF);
		fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
		fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
		fprintf(outfile, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
	}

	// These next 4 characters are for the biPlanes and biBitCount fields.

	fprintf(outfile, "%c", 1);
	fprintf(outfile, "%c", 0);
	fprintf(outfile, "%c", 24);
	fprintf(outfile, "%c", 0);

	for (n = 7; n <= 12; n++)
	{
		fprintf(outfile, "%c", headers[n] & 0x000000FF);
		fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
		fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
		fprintf(outfile, "%c", (headers[n] & (unsigned int) 0xFF000000) >> 24);
	}

	//
	// Headers done, now write the data...
	//

	for (y = HEIGHT - 1; y >= 0; y--)     // BMP image format is written from bottom to top...
	{
		for (x = 0; x <= WIDTH - 1; x++)
		{
			/*
			red = reduce(redcount[x][y] + COLOUR_OFFSET) * red_multiplier;
			green = reduce(greencount[x][y] + COLOUR_OFFSET) * green_multiplier;
			blue = reduce(bluecount[x][y] + COLOUR_OFFSET) * blue_multiplier;
			*/

			if (!IS_GREYSCALE) {
				red   = chan[0 + y*LD*BPP + BPP*x];
				green = chan[1 + y*LD*BPP + BPP*x]; 
				blue  = chan[2 + y*LD*BPP + BPP*x];
			} else {
				red   = chan[0 + y*LD*BPP + BPP*x];
				green = red;
				blue  = red;
			}

			if (red > 255) red = 255; if (red < 0) red = 0;
			if (green > 255) green = 255; if (green < 0) green = 0;
			if (blue > 255) blue = 255; if (blue < 0) blue = 0;
			// Also, it's written in (b,g,r) format...

			fprintf(outfile, "%c", blue);
			fprintf(outfile, "%c", green);
			fprintf(outfile, "%c", red);
		}
		if (extrabytes)      // See above - BMP lines must be of lengths divisible by 4.
		{
			for (n = 1; n <= extrabytes; n++)
			{
				fprintf(outfile, "%c", 0);
			}
		}
	}

	fclose(outfile);
	return;
}


void writePPM(const char * filename, unsigned char *chan, int LD, int WIDTH, int HEIGHT, int BPP, int NUMCOMP)
{
    std::ofstream rOutputStream(filename);
    if (!rOutputStream)
    {
        std::cerr << "Cannot open output file: " << filename << std::endl;
        return;   
    }

    
    
    if( NUMCOMP ==4)
    {
        rOutputStream << "P7\n";
        rOutputStream << "#nvTIFF\n";
        rOutputStream << "WIDTH "<<WIDTH<<"\n";
        rOutputStream << "HEIGHT "<<HEIGHT<<"\n";
        rOutputStream << "DEPTH "<<NUMCOMP<<"\n";
        rOutputStream << "MAXVAL "<<(1<<BPP)-1<<"\n";
        rOutputStream << "TUPLTYPE RGB_ALPHA\n";
        rOutputStream << "ENDHDR\n";
    }
    else
    {
        rOutputStream << "P6\n";
        rOutputStream << "#nvTIFF\n";
        rOutputStream << WIDTH << " " << HEIGHT << "\n";
        rOutputStream << (1<<BPP)-1<<"\n";
    }
    for(int y = 0; y < HEIGHT; y++)
    {
        for(int x = 0; x < WIDTH; x++)
        {
            if( BPP == 8)
            {
				rOutputStream << chan[(y*LD + x)*NUMCOMP];
                rOutputStream << chan[(y*LD + x)*NUMCOMP + 1];
                rOutputStream << chan[(y*LD + x)*NUMCOMP + 2];
                if( NUMCOMP == 4)
                {
                    rOutputStream << chan[(y*LD + x)*NUMCOMP + 3];;
                }
            }
            else
            {
                int pixel_offset = (y * LD *NUMCOMP*2 + (x*NUMCOMP*2 ));
				for( int c = 0; c < NUMCOMP; c++)
				{
					rOutputStream << chan[pixel_offset + 2 * c +1]<<chan[pixel_offset + 2*c];
				}

            }
            
        }
    }
    return;
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
		"\t-r\n"
		"\t--rowsxstrip\n"
		"\t\tSpecifies the number of consecutive rows  to  use  to  divide  the  images  into\n"
		"\t\tstrips.  Each image is divided in strips of the same size (except  possibly  the\n"
		"\t\tlast strip) and then the strips are  compressed  as  independent  byte  streams.\n"
		"\t\tThis option is ignored if -E is not specified.\n"
		"\t\tDefault: 1.\n"
		"\n"
		"\t-s\n"
		"\t--stripalloc\n"
		"\t\tSpecifies the initial estimate of the maximum size  of  compressed  strips.   If\n"
		"\t\tduring compression one or more strips require more  space,  the  compression  is\n"
		"\t\taborted and restarted automatically with a safe estimate. \n"
		"\t\tThis option is ignored if -E is not specified.\n"
		"\t\tDefault: the size, in bytes, of a strip in the uncompressed images.\n" 
		"\n"
		"\t--encode-out\n"
		"\t\tEnables the writing of the compressed  images  to  an  output  TIFF  file named\n"
		"\t\toutFile.tif.\n"
		"\t\tDefualt: disabled.\n",
		pname);

	exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {

	int devId = 0;

	char *fname = NULL;

	int verbose = 0;
	int decWriteOutN = 0;

	int frameBeg = INT_MIN;
	int frameEnd = INT_MAX;
	int decodeRange = 0;

	int memType = NVTIFF_MEM_REG;
	int doH2DFileCopy = 0;

	int doEncode = 0;
	int encRowsPerStrip = 1;
	unsigned long long encStripAllocSize = 0;
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
			case 'm':
				if (optarg[0] != 'r' && optarg[0] != 'p') {
					fprintf(stderr, "Unknown memory type specified (%c)!\n", optarg[0]);
					usage(argv[0]);
				}
				memType = (optarg[0] == 'r') ? NVTIFF_MEM_REG : NVTIFF_MEM_PIN;
				break;
			case 'c':
				doH2DFileCopy = 1;
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
			case 'r':
				encRowsPerStrip = atoi(optarg);
				break;
			case 's':
				encStripAllocSize = atoi(optarg);
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

	if (verbose > 1) {
		nvTiffDumpRaw(fname);
	}

	nvtiffStream_t tiff_stream;
	nvtiffDecoder_t decoder;
	nvtiffFileInfo_t file_info;
    CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
	CHECK_NVTIFF(nvtiffDecoderCreate(&decoder,
        nullptr, nullptr, 0));

    CHECK_NVTIFF(nvtiffStreamParseFromFile(fname, tiff_stream));

    
    CHECK_NVTIFF(nvtiffStreamGetFileInfo(tiff_stream, &file_info));

	// BEGIN work (possibly) overlapped with H2D copy of the file data
	if (verbose) {
		CHECK_NVTIFF(nvtiffStreamPrint(tiff_stream));
	}
	
	frameBeg = fmax(frameBeg, 0);
	frameEnd = fmin(frameEnd, file_info.num_images-1);
	const int nDecode = frameEnd-frameBeg+1;

	// allocate device memory for images
	unsigned char **imageOut_d = NULL;

	const size_t imageSize = sizeof(**imageOut_d)*file_info.image_width *
						      file_info.image_height *
						      (file_info.bits_per_pixel/8);

	imageOut_d = (unsigned char **)Malloc(sizeof(*imageOut_d)*nDecode);
	for(unsigned int i = 0; i < nDecode; i++) {
		CHECK_CUDA(cudaMalloc(imageOut_d+i, imageSize));
	}

	printf("Decoding %u, %s %ux%u images [%d, %d], from file %s... ",
		nDecode,
		file_info.photometric_int == NVTIFF_PHOTOMETRIC_RGB ? "RGB" : "Grayscale",
		file_info.image_width,
		file_info.image_height,
		frameBeg,
		frameEnd,
		fname);
	fflush(stdout);


	double __t = Wtime();
	if (!decodeRange) {
		CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, imageOut_d, stream));
	} else { 
		CHECK_NVTIFF(nvtiffDecodeRange(tiff_stream, decoder, frameBeg, nDecode, imageOut_d, stream));
	}
	CHECK_CUDA(cudaStreamSynchronize(stream));
	__t = Wtime()-__t;



	printf("done in %lf secs\n\n", __t);

	if (decWriteOutN) {

		unsigned char *imageOut_h = (unsigned char *)Malloc(sizeof(*imageOut_h)*imageSize);

		const unsigned int nout = fmin(decWriteOutN, nDecode);

		printf("\tWriting images for the first %d subfile(s)...\n", nout);
		fflush(stdout);

		__t = Wtime();
		for(unsigned int i = 0; i < nout; i++) {

			CHECK_CUDA(cudaMemcpy(imageOut_h, imageOut_d[i], imageSize, cudaMemcpyDeviceToHost));

			char outfname[MAX_STR_LEN];

			const int isgreyScale = (file_info.photometric_int == NVTIFF_PHOTOMETRIC_MINISWHITE) ||
						(file_info.photometric_int == NVTIFF_PHOTOMETRIC_MINISBLACK);
			//void writePPM(const char * filename, unsigned char *chan, int LD, int WIDTH, int HEIGHT, int BPP, int NUMCOMP)

			if(file_info.bits_per_sample[0] == 16)
			{
					snprintf(outfname, MAX_STR_LEN, "outImage_%d.ppm", i);
					 writePPM(outfname,
					     imageOut_h,
					     file_info.image_width,
					     file_info.image_width, 
					     file_info.image_height,
					     file_info.bits_per_sample[0], file_info.samples_per_pixel);
			}
			else
			if (!isgreyScale || (isgreyScale && file_info.bits_per_pixel == 8)) {

				snprintf(outfname, MAX_STR_LEN, "outImage_%d.bmp", i);

				printf("\t\timage %u... BMP format\n", i);
				writeBMPFile(outfname,
					     imageOut_h,
					     file_info.image_width,
					     file_info.image_width, 
					     file_info.image_height,
					     //tiffData->subFiles[i].samplesPerPixel,
					     file_info.bits_per_pixel/8,
					     isgreyScale);
			} else {
				snprintf(outfname, MAX_STR_LEN, "outImage_%d.raw", i);

				printf("\t\timage %u... RAW format\n", i);
				FILE *f = Fopen(outfname, "w");
				Fwrite(imageOut_h, imageSize, 1, f);
				fclose(f);
			}
		}
		__t = Wtime()-__t;
		printf("\t...done in %lf secs\n\n", __t);

		free(imageOut_h);
	}

#ifdef LIBTIFF_TEST
	TIFF* tif = TIFFOpen(fname, "r");
	if (tif) {

		// we alredy know that all subfiles have the same porperties
		uint32_t *raster;
		raster = (uint32_t *)_TIFFmalloc(tiffData->subFiles[0].ncol*tiffData->subFiles[0].nrow * sizeof (uint32_t));

		printf("\tDecoding with libTIFF... "); fflush(stdout);
		double __t = Wtime();
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
		__t = Wtime()-__t;
		printf("done in %lf secs\n\n", __t);

		_TIFFfree(raster);
		TIFFClose(tif);
	}
#endif

	if (doEncode) {
#if 0
		unsigned char *tmp = (unsigned char *)Malloc(imageSize);
		for(int i = 0; i < imageSize; i++) {
			tmp[i] = rand()%256;
		}
		CHECK_CUDA(cudaMemcpy(imageOut_d[0], tmp, imageSize, cudaMemcpyHostToDevice));
		free(tmp);
#endif
		unsigned int nrow              = file_info.image_height;
		unsigned int ncol              = file_info.image_width;
		unsigned int photometricInt    = (unsigned int)file_info.photometric_int;
		unsigned int planarConf        = (unsigned int)file_info.planar_config;
		unsigned short pixelSize       = file_info.bits_per_pixel/8;
		unsigned short samplesPerPixel = file_info.samples_per_pixel;
		unsigned short sampleFormat    = file_info.sample_format[0];

		unsigned short *bitsPerSample = (unsigned short *)Malloc(sizeof(*bitsPerSample)*samplesPerPixel);
		memcpy(bitsPerSample,
		       file_info.bits_per_sample,
		       sizeof(*bitsPerSample)*samplesPerPixel);

		CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
        CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
		tiff_stream = NULL;
		decoder = NULL;

		unsigned int nSubFiles = nDecode;
		unsigned int nStripOut = DIV_UP(nrow, encRowsPerStrip);
		unsigned int totStrips = nSubFiles*nStripOut;

		unsigned long long *stripSize_d = NULL;
		unsigned long long *stripOffs_d = NULL;
		unsigned char      *stripData_d = NULL;

		if (encStripAllocSize <= 0) {
			encStripAllocSize = encRowsPerStrip*ncol*(pixelSize);
		}

		CHECK_CUDA(cudaMalloc(&stripSize_d, sizeof(*stripSize_d)*totStrips));
		CHECK_CUDA(cudaMalloc(&stripOffs_d, sizeof(*stripOffs_d)*totStrips));
		CHECK_CUDA(cudaMalloc(&stripData_d, sizeof(*stripData_d)*totStrips*encStripAllocSize));

		nvTiffEncodeCtx_t *ctx = nvTiffEncodeCtxCreate(devId, nSubFiles, nStripOut);

		printf("Encoding %u, %s %ux%u images using %d rows per strip and %llu bytes per strip... ",
			nDecode,
			photometricInt == 2 ? "RGB" : "Grayscale",
			ncol,
			nrow,
			encRowsPerStrip,
			encStripAllocSize);
		fflush(stdout);
		int rv;
		__t = Wtime();
		do {
			rv = nvTiffEncode(ctx,
					  nrow,
					  ncol,
					  pixelSize,
					  encRowsPerStrip,
					  nSubFiles,
					  imageOut_d,
					  encStripAllocSize,
					  stripSize_d,
					  stripOffs_d,
					  stripData_d,
					  stream);
			if (rv != NVTIFF_ENCODE_SUCCESS) {
				printf("error, while encoding images!\n");
				exit(EXIT_FAILURE);
			}
			rv = nvTiffEncodeFinalize(ctx, stream);
			if (rv != NVTIFF_ENCODE_SUCCESS) {
				if (rv == NVTIFF_ENCODE_COMP_OVERFLOW) {
					printf("overflow, using %llu bytes per strip...", ctx->stripSizeMax);

					// * free ctx mem
					// * reallocate a larger stripData_d buffer
					// * init a new ctx and retry
					// * retry compression
					encStripAllocSize = ctx->stripSizeMax;
					nvTiffEncodeCtxDestroy(ctx);
					CHECK_CUDA(cudaFree(stripData_d));
					CHECK_CUDA(cudaMalloc(&stripData_d, sizeof(*stripData_d)*totStrips*encStripAllocSize));
					ctx = nvTiffEncodeCtxCreate(devId, nSubFiles, nStripOut);
				} else {
					printf("error, while finalizing compressed images!\n");
					exit(EXIT_FAILURE);
				}
			}
		} while(rv == NVTIFF_ENCODE_COMP_OVERFLOW);

		CHECK_CUDA(cudaStreamSynchronize(stream));
		__t = Wtime()-__t;

		printf("done in %lf secs (compr. ratio: %.2lfx)\n\n",
			__t, double(imageSize)*nSubFiles/ctx->stripSizeTot);

		//printf("Total size of compressed strips: %llu bytes\n", ctx->stripSizeTot);

		if (encWriteOut) {
			unsigned long long *stripSize_h = (unsigned long long *)Malloc(sizeof(*stripSize_h)*totStrips);
			CHECK_CUDA(cudaMemcpy(stripSize_h,
					      stripSize_d,
					      sizeof(*stripSize_h)*totStrips,
					      cudaMemcpyDeviceToHost));

			unsigned long long *stripOffs_h = (unsigned long long *)Malloc(sizeof(*stripOffs_h)*totStrips);
			CHECK_CUDA(cudaMemcpy(stripOffs_h,
					      stripOffs_d,
					      sizeof(*stripOffs_h)*totStrips,
					      cudaMemcpyDeviceToHost));

			unsigned char *stripData_h = (unsigned char *)Malloc(sizeof(*stripData_h)*ctx->stripSizeTot);
			CHECK_CUDA(cudaMemcpy(stripData_h,
					      stripData_d,
					      ctx->stripSizeTot,
					      cudaMemcpyDeviceToHost));
#if 0
			FILE *fp = Fopen("stripData.txt", "w");

			size_t stripSize = sizeof(*stripData_h)*encRowsPerStrip*ncol*pixelSize;
			for(unsigned int i = 0; i < nSubFiles; i++) {

				fprintf(fp, "compressed image %d:\n", i);
				for(unsigned int j = 0; j < nStripOut; j++) {

					unsigned long long off = stripOffs_h[i*nStripOut + j];
					unsigned long long len = stripSize_h[i*nStripOut + j];

					fprintf(fp, "\tstrip %5u, size: %6llu bytes (ratio: %5.2lfx), "
						    "fingerprint: %02X %02X %02X %02X ... %02X %02X %02X %02X\n",
						j, len, double(stripSize)/len,
						stripData_h[off + 0],
						stripData_h[off + 1],
						stripData_h[off + 2],
						stripData_h[off + 3],
						stripData_h[off + len-4],
						stripData_h[off + len-3],
						stripData_h[off + len-2],
						stripData_h[off + len-1]);
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
#endif
			printf("\tWriting %u compressed images to TIFF file... ", nDecode); fflush(stdout);
			__t = Wtime();
			nvTiffWriteFile("outFile.tif",
					VER_REG_TIFF,
					nSubFiles,
					nrow,
					ncol,
					encRowsPerStrip,
					samplesPerPixel,
					bitsPerSample,
					photometricInt,
					planarConf,
					stripSize_h,
					stripOffs_h,
					stripData_h,
					sampleFormat);
			__t = Wtime()-__t;
			printf("done in %lf secs\n\n", __t);
		
			free(stripSize_h);
			free(stripOffs_h);
			free(stripData_h);
		}

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

		CHECK_CUDA(cudaFree(stripSize_d));
		CHECK_CUDA(cudaFree(stripOffs_d));
		CHECK_CUDA(cudaFree(stripData_d));
		
		free(bitsPerSample);

		nvTiffEncodeCtxDestroy(ctx);
	}

	// cleanup
	for(unsigned int i = 0; i < nDecode; i++) {
		CHECK_CUDA(cudaFree(imageOut_d[i]));
	}
	free(imageOut_d);

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

