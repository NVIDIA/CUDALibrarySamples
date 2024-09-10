
/* Copyright 2020-2024 NVIDIA Corporation And Affiliates.  All rights reserved.
  * 
  * NOTICE TO LICENSEE: 
  * 
  * The source code and/or documentation ("Licensed Deliverables") are 
  * subject to NVIDIA intellectual property rights under U.S. and 
  * international Copyright laws. 
  * 
  * The Licensed Deliverables contained herein are PROPRIETARY and 
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and 
  * conditions of a form of NVIDIA software license agreement by and 
  * between NVIDIA and Licensee ("License Agreement") or electronically 
  * accepted by Licensee.  Notwithstanding any terms or conditions to 
  * the contrary in the License Agreement, reproduction or disclosure 
  * of the Licensed Deliverables to any third party without the express 
  * written consent of NVIDIA is prohibited. 
  * 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY 
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
  * OF THESE LICENSED DELIVERABLES. 
  * 
  * U.S. Government End Users.  These Licensed Deliverables are a 
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
  * 1995), consisting of "commercial computer software" and "commercial 
  * computer software documentation" as such terms are used in 48 
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
  * U.S. Government End Users acquire the Licensed Deliverables with 
  * only those rights set forth herein. 
  * 
  * Any use of the Licensed Deliverables in individual and commercial 
  * software must include, in the user documentation and internal 
  * comments to the code, the above Disclaimer and U.S. Government End 
  * Users Notice. 
  */

#include <iostream>

#include <string.h>  // strcmpi
#ifndef _WIN64
#include <sys/time.h>  // timings
#include <unistd.h>
#endif
#include <dirent.h>  
#include <sys/stat.h>
#include <sys/types.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <stdio.h>
#include <string.h>
#include <fstream>

#include <nppPlus/nppPlus.h>
#include <npp.h>


#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

// Note:  If you want to view these images we HIGHLY recommend using imagej which is free on the internet and works on most platforms 
//        because it is one of the few image viewing apps that can display 32 bit integer image data.  While it normalizes the data
//        to floating point values for viewing it still provides a good representation of the relative brightness of each label value.  
//        Note that label compression output results in smaller differences between label values making it visually more difficult to detect
//        differences in labeled regions.  If you have an editor that can display hex values you can see what the exact values of
//        each label is, every 4 bytes represents 1 32 bit integer label value.
//
//        The files read and written by this sample app use RAW image format, that is, only the image data itself exists in the files
//        with no image format information.   When viewing RAW files with imagej just enter the image size and bit depth values that
//        are part of the file name when requested by imagej.
//
//        Note that there is a small amount of variability in the number of unique label markers generated from one run to the next by the UF algorithm.

#define NUMBER_OF_IMAGES 5

    Npp8u  * pInputImageDev[NUMBER_OF_IMAGES];
    Npp8u  * pInputImageHost[NUMBER_OF_IMAGES];
    Npp8u  * pUFGenerateLabelsScratchBufferDev[NUMBER_OF_IMAGES];
    Npp8u  * pUFGenerateLabelsScratchBufferHost[NUMBER_OF_IMAGES];
    Npp8u * pFillDev[NUMBER_OF_IMAGES];
    Npp8u * pFillHost[NUMBER_OF_IMAGES];

void tearDown() // Clean up and tear down
{
    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        if (pUFGenerateLabelsScratchBufferDev[j] != 0)
        {
            cudaFree(pUFGenerateLabelsScratchBufferDev[j]);
            pUFGenerateLabelsScratchBufferDev[j] = 0;
        }
        if (pFillDev[j] != 0)
        {
            cudaFree(pFillDev[j]);
            pFillDev[j] = 0;
        }
        if (pInputImageDev[j] != 0)
        {
            cudaFree(pInputImageDev[j]);
            pInputImageDev[j] = 0;
        }
        if (pUFGenerateLabelsScratchBufferHost[j] != 0)
        {
            free(pUFGenerateLabelsScratchBufferHost[j]);
            pUFGenerateLabelsScratchBufferHost[j] = 0;
        }
        if (pFillHost[j] != 0)
        {
            free(pFillHost[j]);
            pFillHost[j] = 0;
        }
        if (pInputImageHost[j] != 0)
        {
            free(pInputImageHost[j]);
            pInputImageHost[j] = 0;
        }
    }
}

const std::string & Path = std::string("../images/");

const std::string & InputFile0 = Path + std::string("RainbowChart_RGB_C3_1024x445_8u.raw");
const std::string & InputFile1 = Path + std::string("SeabedSampler_RGB_C3_675x1024_8u.raw");

const std::string & LabelMarkersOutputFile0 = Path + std::string("RainbowChart_RGB_C3_Fill_8Way_1024x445_Dev_8u.raw");
const std::string & LabelMarkersOutputFile1 = Path + std::string("RainbowChart_RGB_C3_Fill_8Way_Gradient_1024x445_Dev_8u.raw");
const std::string & LabelMarkersOutputFile2 = Path + std::string("RainbowChart_RGB_C3_Fill_8Way_Gradient_Boundary_1024x445_Dev_8u.raw");
const std::string & LabelMarkersOutputFile3 = Path + std::string("SeabedSampler_RGB_C3_Fill_8Way_Range_675x1024_Dev_8u.raw");
const std::string & LabelMarkersOutputFile4 = Path + std::string("SeabedSampler_RGB_C3_Fill_8Way_Range_Boundary_675x1024_Dev_8u.raw");

int 
loadRaw8BitImage(Npp8u * pImage, int nWidth, int nHeight, int N, int nImage)
{
    FILE * bmpFile;
    size_t nSize;

    if (nImage == 0 || nImage == 1 || nImage == 2)
    {
        if (nWidth != 1024 || nHeight != 445) 
            return -1;
        fopen_s(&bmpFile, InputFile0.c_str(), "rb");
    }
    else if (nImage == 3 || nImage == 4)
    {
        if (nWidth != 675 || nHeight != 1024) 
            return -1;
        fopen_s(&bmpFile, InputFile1.c_str(), "rb");
    }
    else
    {
        printf ("Input file load failed.\n");
        return -1;
    }

    if (bmpFile == NULL) 
        return -1;
    nSize = fread(pImage, 1, nWidth * N * nHeight, bmpFile);
    if (nSize < nWidth * nHeight)
    {
        fclose(bmpFile);        
        return -1;
    }
    fclose(bmpFile);

    printf ("Input file load succeeded.\n");

    return 0;
}

int 
main( int argc, char** argv )
{

    int      aGenerateDevLabelsScratchBufferSize[NUMBER_OF_IMAGES];
    int      aGenerateHostLabelsScratchBufferSize[NUMBER_OF_IMAGES];

    cudaError_t cudaError;
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;
	FILE * bmpFile;

    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        pInputImageDev[j] = 0;
        pInputImageHost[j] = 0;
        pUFGenerateLabelsScratchBufferDev[j] = 0;
        pUFGenerateLabelsScratchBufferHost[j] = 0;
        pFillDev[j] = 0;
        pFillHost[j] = 0;
    }

    nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

    cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
    {
        printf("CUDA error: no devices supporting CUDA.\n");
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
    }

    const NppLibraryVersion *libVer   = nppPlusV::nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("CUDA Runtime Version: %d.%d\n\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    cudaError = cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor, 
                                       cudaDevAttrComputeCapabilityMajor, 
                                       nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor, 
                                       cudaDevAttrComputeCapabilityMinor, 
                                       nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);

    cudaDeviceProp oDeviceProperties;

    cudaError = cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId);

    nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

    NppiSize oSizeROI[NUMBER_OF_IMAGES];

    NppiPoint oSeed;
    Npp8u oMinGradient[3];
    Npp8u oMaxGradient[3];
    Npp8u oMinRange[3];
    Npp8u oMaxRange[3];
    Npp8u oMinRangeRed[3];
    Npp8u oMaxRangeRed[3];
    Npp8u oMinRangeBlue[3];
    Npp8u oMaxRangeBlue[3];

    oMinGradient[0] = 52;
    oMinGradient[1] = 52;
    oMinGradient[2] = 52;

    oMaxGradient[0] = 3;
    oMaxGradient[1] = 3;
    oMaxGradient[2] = 3;

    oMinRange[0] = 77;
    oMinRange[1] = 141;
    oMinRange[2] = 141;

    oMaxRange[0] = 140;
    oMaxRange[1] = 250;
    oMaxRange[2] = 250;

    oMinRangeRed[0] = 77;
    oMinRangeRed[1] = 2;
    oMinRangeRed[2] = 2;

    oMaxRangeRed[0] = 255;
    oMaxRangeRed[1] = 100;
    oMaxRangeRed[2] = 100;

    oMinRangeBlue[0] = 24;
    oMinRangeBlue[1] = 24;
    oMinRangeBlue[2] = 62;

    oMaxRangeBlue[0] = 240;
    oMaxRangeBlue[1] = 248;
    oMaxRangeBlue[2] = 251;

    const Npp8u nNewVal = 128;
    const Npp8u nBoundaryVal = 255;
    const Npp8u aNewVals[3] = { 255, 0, 0 };
    const Npp8u aNewGreenVals[3] = { 0, 128, 0 };
    const Npp8u aNewBrownVals[3] = { 128, 100, 0 };
    const Npp8u aBoundaryVals[3] = { 255, 255, 0 };
    const Npp8u aBoundaryPurpleVals[3] = { 255, 0, 255 };

    for (int nImage = 0; nImage < 5; nImage++)
	{
		if (nImage < 3)
		{
			oSizeROI[nImage].width = 1024;
			oSizeROI[nImage].height = 445;
		}
        else if (nImage < 5)
        {
            oSizeROI[nImage].width = 675;
            oSizeROI[nImage].height = 1024;
        }

        int N = 3;

        // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can significantly improve the performance of many NPP functions, 
        // for UF function label markers generation or compression DO NOT USE cudaMallocPitch().  Doing so could result in incorrect output.

        cudaError = cudaMalloc ((void**)&pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void**)&pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pInputImageHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp8u) * N * oSizeROI[nImage].height));
        pFillHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp8u) * N * oSizeROI[nImage].height));

        // Use UF functions throughout this sample.

        nppStatus = nppiFloodFillGetBufferSize(oSizeROI[nImage], &aGenerateDevLabelsScratchBufferSize[nImage]);

        // One at a time image processing

        cudaError = cudaMalloc ((void **)&pUFGenerateLabelsScratchBufferDev[nImage], aGenerateDevLabelsScratchBufferSize[nImage]);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        nppiFloodFillGetBufferSize(oSizeROI[nImage], &aGenerateHostLabelsScratchBufferSize[nImage]);

        pUFGenerateLabelsScratchBufferHost[nImage] = reinterpret_cast<Npp8u *>(malloc(aGenerateHostLabelsScratchBufferSize[nImage]));


        if (loadRaw8BitImage(pInputImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, N, nImage) == 0)
        {
            cudaError = cudaMemcpy2DAsync(pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, pInputImageHost[nImage], 
                                                          oSizeROI[nImage].width * sizeof(Npp8u) * N, oSizeROI[nImage].width * sizeof(Npp8u) * N, oSizeROI[nImage].height, 
                                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);
            // Make a working copy
            cudaError = cudaMemcpy2DAsync(pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, 
                                          pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, oSizeROI[nImage].width * sizeof(Npp8u) * N, oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToDevice, nppStreamCtx.hStream);

            if (nImage < 3)
            {
                oSeed.x = 327;
                oSeed.y = 426;
            }
            else
            {
                oSeed.x = 20;
                oSeed.y = 24;
            }

            NppiConnectedRegion oConnectedRegionInfo;

            if (nImage == 0)
            {
                // Fill all pixels which exacty match seed pixel color.
                nppStatus = nppPlusV::nppiFloodFill_Ctx(NPP_8U, NPP_CH_3, pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, oSeed, aNewVals,
                                                        nppiNormInf, oSizeROI[nImage], &oConnectedRegionInfo, pUFGenerateLabelsScratchBufferDev[nImage], nppStreamCtx);
            }
            else if (nImage == 1)
            {
                // Fill all pixels which have pixel values between seed pixel color - min gradient value to seed pixel value + max gradient value.
                nppStatus = nppPlusV::nppiFloodFillGradient_Ctx(NPP_8U, NPP_CH_3, pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, oSeed, oMinGradient, oMaxGradient, aNewVals,
                                                                nppiNormInf, oSizeROI[nImage], &oConnectedRegionInfo, pUFGenerateLabelsScratchBufferDev[nImage], nppStreamCtx);
            }
            else if (nImage == 2)
            {
                // Fill all pixels which have pixel values between seed pixel color - min gradient value to seed pixel value + max gradient value.
                // Surround that region with a boundary color.
                nppStatus = nppPlusV::nppiFloodFillGradientBoundary_Ctx(NPP_8U, NPP_CH_3, pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, oSeed, oMinGradient, oMaxGradient, aNewVals,
                                                                        aBoundaryVals, nppiNormInf, oSizeROI[nImage], &oConnectedRegionInfo, pUFGenerateLabelsScratchBufferDev[nImage], nppStreamCtx);
            }
            else if (nImage == 3)
            {
                // Fill all pixels near seed which contain color values between minimum range to maximum range.
                nppStatus = nppPlusV::nppiFloodFillRange_Ctx(NPP_8U, NPP_CH_3, pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, oSeed, oMinRangeBlue, oMaxRangeBlue, aNewGreenVals,
                                                             nppiNormInf, oSizeROI[nImage], &oConnectedRegionInfo, pUFGenerateLabelsScratchBufferDev[nImage], nppStreamCtx);
            }
            else if (nImage == 4)
            {
                // Fill all pixels near seed which contain color values between minimum range to maximum range.
                // Surround that region with a boundary color.
                nppStatus = nppPlusV::nppiFloodFillRangeBoundary_Ctx(NPP_8U, NPP_CH_3, pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, oSeed, oMinRangeBlue, oMaxRangeBlue, aNewBrownVals,
                                                                     aBoundaryPurpleVals, nppiNormInf, oSizeROI[nImage], &oConnectedRegionInfo, pUFGenerateLabelsScratchBufferDev[nImage], nppStreamCtx);
            }

			if (nppStatus != NPP_SUCCESS)
			{
			    printf("Image fill failed.\n");
				tearDown();
				return -1;
			}

            cudaError = cudaMemcpy2DAsync(pFillHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N,
                                          pFillDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * N, 
    									  oSizeROI[nImage].width * sizeof(Npp8u) * N, oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post fill cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                fopen_s(&bmpFile, LabelMarkersOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, LabelMarkersOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                fopen_s(&bmpFile, LabelMarkersOutputFile2.c_str(), "wb");
            else if (nImage == 3)
                fopen_s(&bmpFile, LabelMarkersOutputFile3.c_str(), "wb");
            else if (nImage == 4)
                fopen_s(&bmpFile, LabelMarkersOutputFile4.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            size_t nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
			{
				nSize += fwrite(&pFillHost[nImage][j * oSizeROI[nImage].width * N], sizeof(Npp8u), oSizeROI[nImage].width * N, bmpFile);
            }
            fclose(bmpFile);

            printf ("BoundsRect x %d y %d width %d height %d count %d seed0 %d seed1 %d seed2 %d. \n", 
					oConnectedRegionInfo.oBoundingBox.x, oConnectedRegionInfo.oBoundingBox.y, 
					oConnectedRegionInfo.oBoundingBox.width, oConnectedRegionInfo.oBoundingBox.height, 
					oConnectedRegionInfo.nConnectedPixelCount,
					oConnectedRegionInfo.aSeedPixelValue[0], oConnectedRegionInfo.aSeedPixelValue[1], oConnectedRegionInfo.aSeedPixelValue[2]);
        }
        tearDown();
    }

    return 0;
}



