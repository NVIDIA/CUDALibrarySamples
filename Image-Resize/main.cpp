/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
  

#include "main.h"

//#define OPTIMIZED_HUFFMAN
//#define CUDA10U2

// *****************************************************************************
// nvJPEG handles and parameters
// -----------------------------------------------------------------------------
nvjpegBackend_t impl = NVJPEG_BACKEND_DEFAULT;
nvjpegHandle_t nvjpeg_handle;
nvjpegJpegStream_t nvjpeg_jpeg_stream;
nvjpegDecodeParams_t nvjpeg_decode_params;
nvjpegJpegState_t nvjpeg_decoder_state;
nvjpegEncoderParams_t nvjpeg_encode_params;
nvjpegEncoderState_t nvjpeg_encoder_state;

#ifdef CUDA10U2 // This part needs CUDA 10.1 Update 2 for copy the metadata other information from base image.
nvjpegJpegEncoding_t nvjpeg_encoding;
#endif


// *****************************************************************************
// Decode, Resize and Encoder function
// -----------------------------------------------------------------------------
int decodeResizeEncodeOneImage(std::string sImagePath, std::string sOutputPath, double &time, int resize_width, int resize_height, int resize_quality)
{
	// Decode, Encoder format
	nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_RGBI;
    nvjpegInputFormat_t iformat = NVJPEG_INPUT_RGBI;

	// timing for resize
	time = 0.;
    float resize_time = 0.;
	cudaEvent_t start, stop;
	CHECK_CUDA( cudaEventCreate(&start) );
	CHECK_CUDA( cudaEventCreate(&stop) );

	// Image reading section
	// Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = sImagePath.rfind("/");
    std::string sFileName = (std::string::npos == position)? sImagePath : sImagePath.substr(position + 1, sImagePath.size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position)? sFileName : sFileName.substr(0, position);

#ifndef _WIN64
    position = sFileName.rfind("/");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position + 1, sFileName.length());
#else
    position = sFileName.rfind("\\");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position+1, sFileName.length());
#endif

    // Read an image from disk.
    std::ifstream oInputStream(sImagePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if(!(oInputStream.is_open()))
    {
        std::cerr << "Cannot open image: " << sImagePath << std::endl;
        return EXIT_FAILURE;
    }

    // Get the size.
    std::streamsize nSize = oInputStream.tellg();
    oInputStream.seekg(0, std::ios::beg);

    // Image buffers. 
    unsigned char * pBuffer = NULL; 
    unsigned char * pResizeBuffer = NULL;
    
    std::vector<char> vBuffer(nSize);
    if (oInputStream.read(vBuffer.data(), nSize))
    {            
        unsigned char * dpImage = (unsigned char *)vBuffer.data();
        
        // Retrieve the componenet and size info.
        int nComponent = 0;
        nvjpegChromaSubsampling_t subsampling;
        int widths[NVJPEG_MAX_COMPONENT];
        int heights[NVJPEG_MAX_COMPONENT];
        if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights))
        {
            std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
            return EXIT_FAILURE;
        }
		if(resize_width == 0 || resize_height == 0)
		{
			resize_width = widths[0]/2;
			resize_height = heights[0]/2;
		}
		NppiSize dst_size = {(int)resize_width, (int)resize_height};

		size_t pitch;
		pitch = NVJPEG_MAX_COMPONENT * widths[0];
		cudaError_t eCopy = cudaMalloc(&pBuffer, pitch * heights[0]);
		if(cudaSuccess != eCopy)
		{
			std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(eCopy) << std::endl;
			return EXIT_FAILURE;
		}

		// device image buffers.
		nvjpegImage_t imgdesc;
		nvjpegImage_t imgresize;

		imgdesc.channel[0] = pBuffer;
		imgdesc.channel[1] = pBuffer + widths[0] * heights[0];
		imgdesc.channel[2] = pBuffer + widths[0] * heights[0] * 2;
		imgdesc.channel[3] = pBuffer + widths[0] * heights[0] * 3;
		imgdesc.pitch[0] = (unsigned int)(is_interleaved(oformat) ? widths[0] * NVJPEG_MAX_COMPONENT : widths[0]);
		imgdesc.pitch[1] = (unsigned int)widths[0];
		imgdesc.pitch[2] = (unsigned int)widths[0];
		imgdesc.pitch[3] = (unsigned int)widths[0];

		// image resize
		NppiSize src_size = {(int)widths[0], (int)heights[0]};
		NppiRect src_roi = { 0, 0, src_size.width, src_size.height};
		NppiRect dst_roi = { 0, 0, dst_size.width, dst_size.height};

		pitch = NVJPEG_MAX_COMPONENT * resize_width;
		cudaError_t eCopy1 = cudaMalloc(&pResizeBuffer,pitch * resize_height);
		if(cudaSuccess != eCopy1)
		{
			std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(eCopy) << std::endl;
			return EXIT_FAILURE;
		}

		imgresize.channel[0] = pResizeBuffer;
		imgresize.channel[1] = pResizeBuffer + resize_width * resize_height;
		imgresize.channel[2] = pResizeBuffer + resize_width * resize_height * 2;
		imgresize.channel[3] = pResizeBuffer + resize_width * resize_height * 3;
		imgresize.pitch[0] = (unsigned int)(is_interleaved(oformat) ? resize_width * NVJPEG_MAX_COMPONENT : resize_width);;
		imgresize.pitch[1] = (unsigned int)resize_width;
		imgresize.pitch[2] = (unsigned int)resize_width;
		imgresize.pitch[3] = (unsigned int)resize_width;

		// Timing start
		CHECK_CUDA( cudaEventRecord(start, 0) );

#ifdef CUDA10U2 // This part needs CUDA 10.1 Update 2
		//parse image save metadata in jpegStream structure
		CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, dpImage, nSize, 1, 0, nvjpeg_jpeg_stream));
#endif

		int nReturnCode = 0;

		// decode by stages
		nReturnCode = nvjpegDecode(nvjpeg_handle, nvjpeg_decoder_state, dpImage, nSize, oformat, &imgdesc, NULL);

		cudaDeviceSynchronize();
		if(nReturnCode != 0)
		{
			std::cerr << "Error in nvjpegDecode." << std::endl;
			return EXIT_FAILURE;
		}

		// image resize
		/* Note: this is the simplest resizing function from NPP. */
		NppStatus st;
		NppStreamContext nppStreamCtx;
		nppStreamCtx.hStream = NULL; // default stream

		st = nppiResize_8u_C3R_Ctx(imgdesc.channel[0], imgdesc.pitch[0], src_size, src_roi,
				imgresize.channel[0], imgresize.pitch[0], dst_size, dst_roi, NPPI_INTER_LANCZOS, nppStreamCtx);

		if (st != NPP_SUCCESS)
		{
			std::cerr << "NPP resize failed : " << std::endl;
			return EXIT_FAILURE;
		}
		std::cout << "Resize-width: " << dst_size.width << " Resize-height: " << dst_size.height << std::endl;

		// get encoding from the jpeg stream and copy it to the encode parameters
#ifdef CUDA10U2 // This part needs CUDA 10.1 Update 2 for copy the metadata other information from base image.
		CHECK_NVJPEG(nvjpegJpegStreamGetJpegEncoding(nvjpeg_jpeg_stream, &nvjpeg_encoding));
		CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(nvjpeg_encode_params, nvjpeg_encoding, NULL));
		CHECK_NVJPEG(nvjpegEncoderParamsCopyQuantizationTables(nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
		CHECK_NVJPEG(nvjpegEncoderParamsCopyHuffmanTables(nvjpeg_encoder_state, nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
		CHECK_NVJPEG(nvjpegEncoderParamsCopyMetadata(nvjpeg_encoder_state, nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
#endif

		CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nvjpeg_encode_params, resize_quality, NULL));

#ifdef OPTIMIZED_HUFFMAN  // Optimized Huffman
		CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nvjpeg_encode_params, 1, NULL));
#endif
		CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nvjpeg_encode_params, subsampling, NULL));

		cudaDeviceSynchronize();

		// encoding the resize data
		CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle,
			nvjpeg_encoder_state,
			nvjpeg_encode_params,
			&imgresize,
			iformat,
			dst_size.width,
			dst_size.height,
			NULL));

		// retrive the encoded bitstream for file writing
		std::vector<unsigned char> obuffer;
		size_t length;
		CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
			nvjpeg_handle,
			nvjpeg_encoder_state,
			NULL,
			&length,
			NULL));
		obuffer.resize(length);
		CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
			nvjpeg_handle,
			nvjpeg_encoder_state,
			obuffer.data(),
			&length,
			NULL));

		// Timing stop
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );

		// file writing
		std::string output_filename = sOutputPath + "/" + sFileName + ".jpg";
		char directory[120];
		char mkdir_cmd[256];
		std::string folder = sOutputPath;
		output_filename = folder + "/"+ sFileName +".jpg";
#if !defined(_WIN32)
		sprintf(directory, "%s", folder.c_str());
		sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);
#else
		sprintf(directory, "%s", folder.c_str());
		sprintf(mkdir_cmd, "mkdir %s 2> nul", directory);
#endif

		int ret = system(mkdir_cmd);

		std::cout << "Writing JPEG file: " << output_filename << std::endl;
		std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
		outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
    }
	// Free memory
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(pResizeBuffer));

	// get timing
    CHECK_CUDA( cudaEventElapsedTime(&resize_time, start, stop) );
    time = (double)resize_time;

    return EXIT_SUCCESS;
}

// *****************************************************************************
// parsing the arguments function
// -----------------------------------------------------------------------------
int processArgs(image_resize_params_t param)
{
	std::string sInputPath(param.input_dir);
    std::string sOutputPath(param.output_dir);
    int resize_width = param.width;
    int resize_height = param.height;
    int resize_quality = param.quality;

    int error_code = 1;

	double total_time = 0., decode_time = 0.;
    int total_images = 0;

    std::vector<std::string> inputFiles;
    if (readInput(sInputPath, inputFiles))
    {
        return error_code;
    }
    for (unsigned int i = 0; i < inputFiles.size(); i++)
    {
        std::string &sFileName = inputFiles[i];
        std::cout << "Processing file: " << sFileName << std::endl;

		int image_error_code = decodeResizeEncodeOneImage(sFileName, sOutputPath, decode_time, resize_width, resize_height, resize_quality);

		if (image_error_code)
		{
			std::cerr << "Error processing file: " << sFileName << std::endl;
			return image_error_code;
		}
        else
        {
            total_images++;
            total_time += decode_time;
        }
    }

	std::cout << "------------------------------------------------------------- " << std::endl;
    std::cout << "Total images resized: " << total_images << std::endl;
    std::cout << "Total time spent on resizing: " << total_time << " (ms)" << std::endl;
    std::cout << "Avg time/image: " << total_time/total_images << " (ms)" << std::endl;
	std::cout << "------------------------------------------------------------- " << std::endl;
    return EXIT_SUCCESS;
}

// *****************************************************************************
// main image resize function
// -----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{
	int pidx;

	if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
	(pidx = findParamIndex(argv, argc, "--help")) != -1) {
		std::cout << "Usage: " << argv[0]
		  << " -i images-dir  [-o output-dir]"
			 "[-q jpeg-quality][-rw resize-width ] [-rh resize-height]\n";
		std::cout << "Parameters: " << std::endl;
		std::cout << "\timages-dir\t:\tPath to single image or directory of images" << std::endl;
		std::cout << "\toutput-dir\t:\tWrite resized images to this directory [default resize_output]" << std::endl;
		std::cout << "\tJPEG Quality\t:\tUse image quality [default 85]" << std::endl;
		std::cout << "\tResize Width\t:\t Resize width [default original_img_width/2]" << std::endl;
		std::cout << "\tResize Height\t:\t Resize height [default original_img_height/2]" << std::endl;
		return EXIT_SUCCESS;
	}

	image_resize_params_t params;

	params.input_dir = "./";
	if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
	params.input_dir = argv[pidx + 1];
	} else {
	// Search in default paths for input images.
	int found = getInputDir(params.input_dir, argv[0]);
	if (!found)
	{
	  std::cout << "Please specify input directory for image resizing"<< std::endl;
	  return EXIT_FAILURE;
	}
	}
	if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
	params.output_dir = argv[pidx + 1];
	} else {
	  // by-default write the folder named "output" in cwd
	  params.output_dir = "resize_output";
	}

	params.quality = 85;
	if ((pidx = findParamIndex(argv, argc, "-q")) != -1) {
	params.quality = std::atoi(argv[pidx + 1]);
	}

	params.width = 0;
	if ((pidx = findParamIndex(argv, argc, "-rw")) != -1) {
	params.width = std::atoi(argv[pidx + 1]);
	}

	params.height = 0;
	if ((pidx = findParamIndex(argv, argc, "-rh")) != -1) {
	params.height = std::atoi(argv[pidx + 1]);
	}

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    CHECK_NVJPEG(nvjpegCreate(impl, &dev_allocator, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_decoder_state));

    // create bitstream object
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &nvjpeg_jpeg_stream));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &nvjpeg_encoder_state, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &nvjpeg_encode_params, NULL));

    pidx = processArgs(params);

    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(nvjpeg_encode_params));
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(nvjpeg_encoder_state));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoder_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    
    return pidx;
}
