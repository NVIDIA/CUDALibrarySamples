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

#include <cuda_runtime.h>
#include "nvTiff_utils.h"
#include "cudamacro.h"
#include <cuda_runtime_api.h>
#include <nvTiff.h>

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

/*
static void usage(const char *pname) {
	
	const char *bname = rindex(pname, '/');
	if (!bname) bname = pname;
	else	    bname++;

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
		bname);

	exit(EXIT_FAILURE);
}
*/
int main(int argc, char **argv) {

	int devId = 0;

	char *fname = NULL;

	int verbose = 0;
	int decWriteOutN = 0;

	int frameBeg = INT_MIN;
	int frameEnd = INT_MAX;
	int decodeRange = 0;

	int memType = nvTiff_MEM_REG;
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
					//usage(argv[0]);
				}
				memType = (optarg[0] == 'r') ? nvTiff_MEM_REG : nvTiff_MEM_PIN;
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
				//usage(argv[0]);
			default:
				fprintf(stderr, "unknown option: %c\n", och);
				//usage(argv[0]);
		}
	}

	if (!fname) {
		fprintf(stderr, "Please specify a TIFF file with the -f option!\n");
		//usage(argv[0]);
	}

	if (frameBeg > frameEnd) {
		fprintf(stderr, "Invalid frame range!\n");
		//usage(argv[0]);
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

	nvTiffFile_t *tiffData = nvTiffOpen(devId, fname, memType);
	if (!tiffData) {
		fprintf(stderr, "Error while reading file %s\n", fname);
		exit(EXIT_FAILURE);
	}
	
	if (doH2DFileCopy) {
		// overlap H2D copy of file data with something else...
		nvTiffH2DAsync(tiffData, stream);
	}

	// BEGIN work (possibly) overlapped with H2D copy of the file data
	if (verbose) {
		nvTiffPrint(tiffData);
	}
	
	frameBeg = fmax(frameBeg, 0);
	frameEnd = fmin(frameEnd, tiffData->nSubFiles-1);
	const int nDecode = frameEnd-frameBeg+1;

	// allocate device memory for images
	unsigned char **imageOut_d = NULL;

	const size_t imageSize = sizeof(**imageOut_d)*tiffData->subFiles[0].nrow*
						      tiffData->subFiles[0].ncol*
						      (tiffData->subFiles[0].bitsPerPixel/8);

	imageOut_d = (unsigned char **)Malloc(sizeof(*imageOut_d)*nDecode);
	for(unsigned int i = 0; i < nDecode; i++) {
		CHECK_CUDA(cudaMalloc(imageOut_d+i, imageSize));
	}

	printf("Decoding %u, %s %ux%u images [%d, %d], from file %s... ",
		nDecode,
		tiffData->subFiles[0].photometricInt == 2 ? "RGB" : "Grayscale",
		tiffData->subFiles[0].ncol,
		tiffData->subFiles[0].nrow,
		frameBeg,
		frameEnd,
		fname);
	fflush(stdout);
	// END work (possibly) overlapped with H2D copy of the file data

	if (doH2DFileCopy) {
		// synchronize on the H2D file data copy;
		// not necessary, done only to avoid including
		// data copy in kernel time
		CHECK_CUDA(cudaStreamSynchronize(stream));
	}

	int rv;

	double __t = Wtime();
	if (!decodeRange) {
		rv = nvTiffDecode(tiffData, imageOut_d, stream);
	} else { 
		rv = nvTiffDecodeRange(tiffData, frameBeg, nDecode, imageOut_d, stream);
	}
	CHECK_CUDA(cudaStreamSynchronize(stream));
	__t = Wtime()-__t;

	if (rv != nvTiff_DECODE_SUCCESS) {
		printf("error, while decoding file!\n");
		exit(EXIT_FAILURE);
	}

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

			const int isgreyScale = (tiffData->subFiles[i].photometricInt == 1) ||
						(tiffData->subFiles[i].photometricInt == 0);

			if (!isgreyScale || (isgreyScale && tiffData->subFiles[i].bitsPerPixel == 8)) {

				snprintf(outfname, MAX_STR_LEN, "outImage_%d.bmp", i);

				printf("\t\timage %u... BMP format\n", i);
				writeBMPFile(outfname,
					     imageOut_h,
					     tiffData->subFiles[i].ncol,
					     tiffData->subFiles[i].ncol, 
					     tiffData->subFiles[i].nrow,
					     //tiffData->subFiles[i].samplesPerPixel,
					     tiffData->subFiles[i].bitsPerPixel/8,
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


	if (doEncode) {
		unsigned int nrow              = tiffData->subFiles[0].nrow;
		unsigned int ncol              = tiffData->subFiles[0].ncol;
		unsigned int photometricInt    = tiffData->subFiles[0].photometricInt;
		unsigned int planarConf        = tiffData->subFiles[0].planarConf;
		unsigned short pixelSize       = tiffData->subFiles[0].bitsPerPixel/8;
		unsigned short samplesPerPixel = tiffData->subFiles[0].samplesPerPixel;

		unsigned short *bitsPerSample = (unsigned short *)Malloc(sizeof(*bitsPerSample)*samplesPerPixel);
		memcpy(bitsPerSample,
		       tiffData->subFiles[0].bitsPerSample,
		       sizeof(*bitsPerSample)*samplesPerPixel);

		// free device memory containing TIFF file
		// used for decoding... 
		nvTiffClose(tiffData);
		tiffData = NULL;

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
			if (rv != nvTiff_ENCODE_SUCCESS) {
				printf("error, while encoding images!\n");
				exit(EXIT_FAILURE);
			}
			rv = nvTiffEncodeFinalize(ctx, stream);
			if (rv != nvTiff_ENCODE_SUCCESS) {
				if (rv == nvTiff_ENCODE_COMP_OVERFLOW) {
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
		} while(rv == nvTiff_ENCODE_COMP_OVERFLOW);

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
					stripData_h);
			__t = Wtime()-__t;
			printf("done in %lf secs\n\n", __t);
		
			free(stripSize_h);
			free(stripOffs_h);
			free(stripData_h);
		}

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

	if (tiffData) {
		nvTiffClose(tiffData);
	}
	
	CHECK_CUDA(cudaStreamDestroy(stream));

	CHECK_CUDA(cudaDeviceReset());

	return 0;
}

