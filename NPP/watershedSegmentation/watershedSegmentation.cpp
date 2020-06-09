/* Copyright 2020 NVIDIA Corporation.  All rights reserved.
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

#include "watershedSegmentation.h"

// Note:  If you want to view these images we HIGHLY recommend using imagej which is free on the internet and works on most platforms 
//        because it is one of the few image viewing apps that can display 32 bit integer image data.  While it normalizes the data
//        to floating point values for viewing it still provides a good representation of the relative brightness of each label value.  
//
//        The files read and written by this sample app use RAW image format, that is, only the image data itself exists in the files
//        with no image format information.   When viewing RAW files with imagej just enter the image size and bit depth values that
//        are part of the file name when requested by imagej.
//

#define NUMBER_OF_IMAGES 5

    Npp8u  * pInputImageDev[NUMBER_OF_IMAGES];
    Npp8u  * pInputImageHost[NUMBER_OF_IMAGES];
    Npp8u  * pSegmentationScratchBufferDev[NUMBER_OF_IMAGES];
    Npp8u * pSegmentsDev[NUMBER_OF_IMAGES];
    Npp8u * pSegmentsHost[NUMBER_OF_IMAGES];
    Npp32u * pSegmentLabelsOutputBufferDev[NUMBER_OF_IMAGES];
    Npp32u * pSegmentLabelsOutputBufferHost[NUMBER_OF_IMAGES];

void tearDown() // Clean up and tear down
{
    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        if (pSegmentLabelsOutputBufferDev[j] != 0)
            cudaFree(pSegmentLabelsOutputBufferDev[j]);
        if (pSegmentationScratchBufferDev[j] != 0)
            cudaFree(pSegmentationScratchBufferDev[j]);
        if (pSegmentsDev[j] != 0)
            cudaFree(pSegmentsDev[j]);
        if (pInputImageDev[j] != 0)
            cudaFree(pInputImageDev[j]);
        if (pSegmentLabelsOutputBufferHost[j] != 0)
            free(pSegmentLabelsOutputBufferHost[j]);
        if (pSegmentsHost[j] != 0)
            free(pSegmentsHost[j]);
        if (pInputImageHost[j] != 0)
            free(pInputImageHost[j]);
    }
}

const std::string & Path = std::string("../images/");

const std::string & InputFile0 = Path + std::string("Lena_512x512_8u_Gray.raw");
const std::string & InputFile1 = Path + std::string("CT_skull_512x512_8u_Gray.raw");
const std::string & InputFile2 = Path + std::string("Rocks_512x512_8u_Gray.raw");
const std::string & InputFile3 = Path + std::string("coins_500x383_8u_Gray.raw");
const std::string & InputFile4 = Path + std::string("coins_overlay_500x569_8u_Gray.raw");

const std::string & SegmentsOutputFile0 = Path + std::string("Lena_Segments_8Way_512x512_8u.raw");
const std::string & SegmentsOutputFile1 = Path + std::string("CT_skull_Segments_8Way_512x512_8u.raw");
const std::string & SegmentsOutputFile2 = Path + std::string("Rocks_Segments_8Way_512x512_8u.raw");
const std::string & SegmentsOutputFile3 = Path + std::string("coins_Segments_8Way_500x383_8u.raw");
const std::string & SegmentsOutputFile4 = Path + std::string("coins_overlay_segments_500x569_8u.raw");

const std::string & SegmentBoundariesOutputFile0 = Path + std::string("Lena_SegmentBoundaries_8Way_512x512_8u.raw");
const std::string & SegmentBoundariesOutputFile1 = Path + std::string("CT_skull_SegmentBoundaries_8Way_512x512_8u.raw");
const std::string & SegmentBoundariesOutputFile2 = Path + std::string("Rocks_SegmentBoundaries_8Way_512x512_8u.raw");
const std::string & SegmentBoundariesOutputFile3 = Path + std::string("coins_SegmentBoundaries_8Way_500x383_8u.raw");
const std::string & SegmentBoundariesOutputFile4 = Path + std::string("coins_overlay_SegmentBoundaries_8Way_500x569_8u.raw");

const std::string & SegmentsWithContrastingBoundariesOutputFile0 = Path + std::string("Lena_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw");
const std::string & SegmentsWithContrastingBoundariesOutputFile1 = Path + std::string("CT_skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw");
const std::string & SegmentsWithContrastingBoundariesOutputFile2 = Path + std::string("Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u.raw");
const std::string & SegmentsWithContrastingBoundariesOutputFile3 = Path + std::string("coins_SegmentsWithContrastingBoundaries_8Way_500x383_8u.raw");
const std::string & SegmentsWithContrastingBoundariesOutputFile4 = Path + std::string("coins_overlay_SegmentsWithContrastingBoundaries_8Way_500x569_8u.raw");

const std::string & CompressedSegmentLabelsOutputFile0 = Path + std::string("Lena_CompressedSegmentLabels_8Way_512x512_32u.raw");
const std::string & CompressedSegmentLabelsOutputFile1 = Path + std::string("CT_skull_CompressedSegmentLabels_8Way_512x512_32u.raw");
const std::string & CompressedSegmentLabelsOutputFile2 = Path + std::string("Rocks_CompressedSegmentLabels_8Way_512x512_32u.raw");
const std::string & CompressedSegmentLabelsOutputFile3 = Path + std::string("coins_CompressedSegmentLabels_8Way_500x383_32u.raw");
const std::string & CompressedSegmentLabelsOutputFile4 = Path + std::string("coins_overlay_CompressedSegmentLabels_8Way_500x569_32u.raw");

int 
loadRaw8BitImage(Npp8u * pImage, int nWidth, int nHeight, int nImage)
{
    FILE * bmpFile;
    size_t nSize;

    if (nImage == 0)
    {
        if (nWidth != 512 || nHeight != 512) 
            return -1;
        fopen_s(&bmpFile, InputFile0.c_str(), "rb");
    }
    else if (nImage == 1)
    {
        if (nWidth != 512 || nHeight != 512) 
            return -1;
        fopen_s(&bmpFile, InputFile1.c_str(), "rb");
    }
    else if (nImage == 2)
    {
        if (nWidth != 512 || nHeight != 512) 
            return -1;
        fopen_s(&bmpFile, InputFile2.c_str(), "rb");
    }
    else if (nImage == 3)
    {
        if (nWidth != 500 || nHeight != 383) 
            return -1;
        fopen_s(&bmpFile, InputFile3.c_str(), "rb");
    }
    else if (nImage == 4)
    {
        if (nWidth != 500 || nHeight != 569) 
            return -1;
        fopen_s(&bmpFile, InputFile4.c_str(), "rb");
    }
    else
    {
        printf ("Input file load failed.\n");
        return -1;
    }

    if (bmpFile == NULL)
    {
        printf ("Input file load failed.\n");
        return -1;
    }
    nSize = fread(pImage, 1, nWidth * nHeight, bmpFile);
    if (nSize < nWidth * nHeight)
    {
        printf ("Input file load failed.\n");
        fclose(bmpFile);        
        return -1;
    }
    fclose(bmpFile);

    printf ("Input file load succeeded.\n");

    return 0;
}

// *****************************************************************************
// main watershed segmentation example function
// -----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
    (pidx = findParamIndex(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0]
          << "[-b number-of-batch]\n";
        std::cout << "Parameters: " << std::endl;
        std::cout << "\tnumber-of-batch\t:\tUse number of batch to process [default 3]" << std::endl;
        return EXIT_SUCCESS;
    }

    image_watershedsegmentation_params_t params;

    params.numofbatch = 3;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
    params.numofbatch = std::atoi(argv[pidx + 1]);
    }

    int      aSegmentationScratchBufferSize[NUMBER_OF_IMAGES];
    int      aSegmentLabelsOutputBufferSize[NUMBER_OF_IMAGES];

    cudaError_t cudaError;
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;
    FILE * bmpFile;
    NppiNorm eNorm = nppiNormInf; // default to 8 way neighbor search

    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        pInputImageDev[j] = 0;
        pInputImageHost[j] = 0;
        pSegmentationScratchBufferDev[j] = 0;
        pSegmentLabelsOutputBufferDev[j] = 0;
        pSegmentLabelsOutputBufferHost[j] = 0;
        pSegmentsDev[j] = 0;
        pSegmentsHost[j] = 0;
    }

    nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

    cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
    {
        printf("CUDA error: no devices supporting CUDA.\n");
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
    }

    const NppLibraryVersion *libVer   = nppGetLibVersion();

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

    for (int nImage = 0; nImage < params.numofbatch; nImage++)
    {
        if (nImage == 0)
        {
            oSizeROI[nImage].width = 512;
            oSizeROI[nImage].height = 512;
        }
        else if (nImage == 1)
        {
            oSizeROI[nImage].width = 512;
            oSizeROI[nImage].height = 512;
        }
        else if (nImage == 2)
        {
            oSizeROI[nImage].width = 512;
            oSizeROI[nImage].height = 512;
        }
        else if (nImage == 3)
        {
            oSizeROI[nImage].width = 500;
            oSizeROI[nImage].height = 383;
        }
        else if (nImage == 4)
        {
            oSizeROI[nImage].width = 500;
            oSizeROI[nImage].height = 569;
        }
        // cudaMallocPitch OR cudaMalloc can be used here, in this sample case width == pitch.

        cudaError = cudaMalloc ((void**)&pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void**)&pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pInputImageHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height));
        pSegmentsHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height));

        nppStatus = nppiSegmentWatershedGetBufferSize_8u_C1R(oSizeROI[nImage], &aSegmentationScratchBufferSize[nImage]);

        cudaError = cudaMalloc ((void **)&pSegmentationScratchBufferDev[nImage], aSegmentationScratchBufferSize[nImage]);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        // Output label marker buffers are only needed if you want to same the generated segmentation labels, they ARE compatible with NPP UF generated labels.
        // Requesting segmentation output may slightly decrease segmentation function performance.  Regardless of the pitch of the segmentation image
        // the segment labels output buffer will have a pitch of oSizeROI[nImage].width * sizeof(Npp32u).

        aSegmentLabelsOutputBufferSize[nImage] = oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height;
        
        cudaError = cudaMalloc ((void **)&pSegmentLabelsOutputBufferDev[nImage], aSegmentLabelsOutputBufferSize[nImage]);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pSegmentLabelsOutputBufferHost[nImage] = reinterpret_cast<Npp32u *>(malloc(oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height));

        if (loadRaw8BitImage(pInputImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, nImage) == 0)
        {
            cudaError = cudaMemcpy2DAsync(pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage], 
                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, 
                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // Make a second copy of the unaltered input image since this function works in place and we want to reuse the input image multiple times.
            cudaError = cudaMemcpy2DAsync(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage], 
                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, 
                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            nppStatus = nppiSegmentWatershed_8u_C1IR_Ctx(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 
                                                         pSegmentLabelsOutputBufferDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), eNorm, 
                                                         NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, oSizeROI[nImage], pSegmentationScratchBufferDev[nImage], nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("Lena segments 8Way 512x512 8u failed.\n");
                else if (nImage == 1)
                    printf("CT skull segments 8Way 512x512 8u failed.\n");
                else if (nImage == 2)
                    printf("Rocks segments 8Way 512x512 8u failed.\n");
                else if (nImage == 3)
                    printf("coins segments 8Way 500x383 8u failed.\n");
                else if (nImage == 4)
                    printf("coins overlay segments 8Way 500x569 8u failed.\n");
                tearDown();
                return -1;
            }

            // Now compress the label markers output to make them easier to view.
            int nCompressedLabelsScratchBufferSize;
            Npp8u * pCompressedLabelsScratchBufferDev;

            nppStatus = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(oSizeROI[nImage].width * oSizeROI[nImage].height, &nCompressedLabelsScratchBufferSize);
            if (nppStatus != NPP_NO_ERROR)
                return nppStatus;

            cudaError = cudaMalloc ((void **)&pCompressedLabelsScratchBufferDev, nCompressedLabelsScratchBufferSize);
            if (cudaError != cudaSuccess)
                return NPP_MEMORY_ALLOCATION_ERR;

            int nCompressedLabelCount = 0;

            nppStatus = nppiCompressMarkerLabelsUF_32u_C1IR(pSegmentLabelsOutputBufferDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage], 
                                                            oSizeROI[nImage].width * oSizeROI[nImage].height, &nCompressedLabelCount, 
                                                            pCompressedLabelsScratchBufferDev);

            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("Lena_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 1)
                    printf("CT_Skull_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 2)
                    printf("Rocks_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 3)
                    printf("coins_CompressedLabelMarkersUF_8Way_500x383_32u failed.\n");
                else if (nImage == 4)
                    printf("coins_CompressedLabelMarkersUF_8Way_500x569_32u failed.\n");
                tearDown();
                return -1;
            }

            // Copy segmented image to host
            cudaError = cudaMemcpy2DAsync(pSegmentsHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 
                                          pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Copy segment labels image to host
            cudaError = cudaMemcpy2DAsync(pSegmentLabelsOutputBufferHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u), 
                                          pSegmentLabelsOutputBufferDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post segmentation cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            // Free single image scratch buffer
            cudaFree(pCompressedLabelsScratchBufferDev);

            // Save default segments file.
            if (nImage == 0)
                fopen_s(&bmpFile, SegmentsOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, SegmentsOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                fopen_s(&bmpFile, SegmentsOutputFile2.c_str(), "wb");
            else if (nImage == 3)
                fopen_s(&bmpFile, SegmentsOutputFile3.c_str(), "wb");
            else if (nImage == 4)
                fopen_s(&bmpFile, SegmentsOutputFile4.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            size_t nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentsHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp8u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_Segments_8Way_512x512_8u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_Segments_8Way_512x512_8u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_Segments_8Way_512x512_8u succeeded.\n");
            else if (nImage == 3)
                printf("coins_Segments_8Way_500x383_8u succeeded.\n");
            else if (nImage == 4)
                printf("coins_overlay_Segments_8Way_500x569_8u succeeded.\n");

            // Save segment labels file.
            if (nImage == 0)
                fopen_s(&bmpFile, CompressedSegmentLabelsOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, CompressedSegmentLabelsOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                fopen_s(&bmpFile, CompressedSegmentLabelsOutputFile2.c_str(), "wb");
            else if (nImage == 3)
                fopen_s(&bmpFile, CompressedSegmentLabelsOutputFile3.c_str(), "wb");
            else if (nImage == 4)
                fopen_s(&bmpFile, CompressedSegmentLabelsOutputFile4.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentLabelsOutputBufferHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_CompressedSegmentLabels_8Way_512x512_32u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_CompressedSegmentLabels_8Way_512x512_32u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_CompressedSegmentLabels_8Way_512x512_32u succeeded.\n");
            else if (nImage == 3)
                printf("coins_CompressedSegmentLabels_8Way_500x383_32u succeeded.\n");
            else if (nImage == 4)
                printf("coins_overlay_CompressedSegmentLabels_8Way_500x569_32u succeeded.\n");

            // Now generate a segment boundaries only output image

            // Make a second copy of the unaltered input image since this function works in place and we want to reuse the input image multiple times.
            cudaError = cudaMemcpy2DAsync(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage], 
                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, 
                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // We already generated segment labels images to skip that this time
            nppStatus = nppiSegmentWatershed_8u_C1IR_Ctx(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 
                                                         0, 0, eNorm, 
                                                         NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY, oSizeROI[nImage], pSegmentationScratchBufferDev[nImage], nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("Lena segment boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 1)
                    printf("CT skull segment boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 2)
                    printf("Rocks segment boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 3)
                    printf("coins segment boundaries 8Way 500x383 8u failed.\n");
                else if (nImage == 4)
                    printf("coins overlay segment boundaries 8Way 500x569 8u failed.\n");
                    
                tearDown();

                return -1;
            }

            // Copy segment boundaries image to host
            cudaError = cudaMemcpy2DAsync(pSegmentsHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 
                                          pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post segmentation cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                fopen_s(&bmpFile, SegmentBoundariesOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, SegmentBoundariesOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                fopen_s(&bmpFile, SegmentBoundariesOutputFile2.c_str(), "wb");
            else if (nImage == 3)
                fopen_s(&bmpFile, SegmentBoundariesOutputFile3.c_str(), "wb");
            else if (nImage == 4)
                fopen_s(&bmpFile, SegmentBoundariesOutputFile4.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentsHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp8u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_SegmentBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_SegmentBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_SegmentBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 3)
                printf("coins_SegmentBoundaries_8Way_500x383_8u succeeded.\n");
            else if (nImage == 4)
                printf("coins_overlay_SegmentBoundaries_8Way_500x569_8u succeeded.\n");

            // Now generate a segmented with contrasting boundaries output image

            // Make a second copy of the unaltered input image since this function works in place and we want to reuse the input image multiple times.
            cudaError = cudaMemcpy2DAsync(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage], 
                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, 
                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // We already generated segment labels images to skip that this time
            nppStatus = nppiSegmentWatershed_8u_C1IR_Ctx(pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 
                                                         0, 0, eNorm, 
                                                         NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST, oSizeROI[nImage], pSegmentationScratchBufferDev[nImage], nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("Lena segments with contrasting boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 1)
                    printf("CT skull segments with contrasting boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 2)
                    printf("Rocks segments with contrasting boundaries 8Way 512x512 8u failed.\n");
                else if (nImage == 3)
                    printf("coins segments with contrasting boundaries 8Way 500x383 8u failed.\n");
                else if (nImage == 4)
                    printf("coins overlay segments with contrasting boundaries 8Way 500x569 8u failed.\n");
                tearDown();
                return -1;
            }

            // Copy segment boundaries image to host
            cudaError = cudaMemcpy2DAsync(pSegmentsHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 
                                          pSegmentsDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post segmentation cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                fopen_s(&bmpFile, SegmentsWithContrastingBoundariesOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, SegmentsWithContrastingBoundariesOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                fopen_s(&bmpFile, SegmentsWithContrastingBoundariesOutputFile2.c_str(), "wb");
            else if (nImage == 3)
                fopen_s(&bmpFile, SegmentsWithContrastingBoundariesOutputFile3.c_str(), "wb");
            else if (nImage == 4)
                fopen_s(&bmpFile, SegmentsWithContrastingBoundariesOutputFile4.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSegmentsHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp8u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 1)
                printf("CT_Skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 2)
                printf("Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.\n");
            else if (nImage == 3)
                printf("coins_SegmentsWithContrastingBoundaries_8Way_500x383_8u succeeded.\n");
            else if (nImage == 4)
                printf("coins_overlay_SegmentsWithContrastingBoundaries_8Way_500x569_8u succeeded.\n");
        }
    }

    tearDown();

    return 0;
}



