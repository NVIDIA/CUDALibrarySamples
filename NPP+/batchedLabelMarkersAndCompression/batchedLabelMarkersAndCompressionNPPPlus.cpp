/* Copyright 2021-2024 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

#include "batchedLabelMarkersAndCompression.h"
#include <nppPlus/nppPlus.h>
#include <npp.h>

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
//        This sample app works in 2 stages, first it processes all of the images individually then it processes them all again in 1 batch
//        using the Batch_Advanced versions of the NPP batch functions which allow each image to have it's own ROI.  The 2 stages are completely
//        separable but in this sample the second stage takes advantage of some of the data that has already been initialized.
//
//        Note that there is a small amount of variability in the number of unique label markers generated from one run to the next by the UF algorithm.
//
//        Performance of ALL NPP image batch functions is limited by the maximum ROI height in the list of images.

// Batched label compression support is only available on NPP versions > 11.0, comment out if using NPP 11.0
//#define CUDA11U1

#define NUMBER_OF_IMAGES 5

    Npp8u  * pInputImageDev[NUMBER_OF_IMAGES];
    Npp8u  * pInputImageHost[NUMBER_OF_IMAGES];
    Npp8u  * pUFGenerateLabelsScratchBufferDev[NUMBER_OF_IMAGES];
    Npp8u  * pUFCompressedLabelsScratchBufferDev[NUMBER_OF_IMAGES];
    Npp32u * pUFLabelDev[NUMBER_OF_IMAGES];
    Npp32u * pUFLabelHost[NUMBER_OF_IMAGES];
    NppiImageDescriptor  * pUFBatchSrcImageListDev = 0;
    NppiImageDescriptor  * pUFBatchSrcDstImageListDev = 0;
    NppiImageDescriptor  * pUFBatchSrcImageListHost = 0;
    NppiImageDescriptor  * pUFBatchSrcDstImageListHost = 0;
    NppiBufferDescriptor * pUFBatchSrcDstScratchBufferListDev = 0; // from nppi_filtering_functions.h
    NppiBufferDescriptor * pUFBatchSrcDstScratchBufferListHost = 0;
    Npp32u * pUFBatchPerImageCompressedCountListDev = 0;
    Npp32u * pUFBatchPerImageCompressedCountListHost = 0;

void tearDown() // Clean up and tear down
{
    if (pUFBatchPerImageCompressedCountListDev != 0)
        cudaFree(pUFBatchPerImageCompressedCountListDev);
    if (pUFBatchSrcDstScratchBufferListDev != 0)
        cudaFree(pUFBatchSrcDstScratchBufferListDev);
    if (pUFBatchSrcDstImageListDev != 0)
        cudaFree(pUFBatchSrcDstImageListDev);
    if (pUFBatchSrcImageListDev != 0)
        cudaFree(pUFBatchSrcImageListDev);
    if (pUFBatchPerImageCompressedCountListHost != 0)
        free(pUFBatchPerImageCompressedCountListHost);
    if (pUFBatchSrcDstScratchBufferListHost != 0)
        free(pUFBatchSrcDstScratchBufferListHost);
    if (pUFBatchSrcDstImageListHost != 0)
        free(pUFBatchSrcDstImageListHost);
    if (pUFBatchSrcImageListHost != 0)
        free(pUFBatchSrcImageListHost);

    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        if (pUFCompressedLabelsScratchBufferDev[j] != 0)
            cudaFree(pUFCompressedLabelsScratchBufferDev[j]);
        if (pUFGenerateLabelsScratchBufferDev[j] != 0)
            cudaFree(pUFGenerateLabelsScratchBufferDev[j]);
        if (pUFLabelDev[j] != 0)
            cudaFree(pUFLabelDev[j]);
        if (pInputImageDev[j] != 0)
            cudaFree(pInputImageDev[j]);
        if (pUFLabelHost[j] != 0)
            free(pUFLabelHost[j]);
        if (pInputImageHost[j] != 0)
            free(pInputImageHost[j]);
    }
}

const std::string & Path = std::string("../images/");

const std::string & InputFile0 = Path + std::string("lena_512x512_8u.raw");
const std::string & InputFile1 = Path + std::string("CT_skull_512x512_8u.raw");
const std::string & InputFile2 = Path + std::string("PCB_METAL_509x335_8u.raw");
const std::string & InputFile3 = Path + std::string("PCB2_1024x683_8u.raw");
const std::string & InputFile4 = Path + std::string("PCB_1280x720_8u.raw");

const std::string & LabelMarkersOutputFile0 = Path + std::string("Lena_LabelMarkersUF_8Way_512x512_32u.raw");
const std::string & LabelMarkersOutputFile1 = Path + std::string("CT_skull_LabelMarkersUF_8Way_512x512_32u.raw");
const std::string & LabelMarkersOutputFile2 = Path + std::string("PCB_METAL_LabelMarkersUF_8Way_509x335_32u.raw");
const std::string & LabelMarkersOutputFile3 = Path + std::string("PCB2_LabelMarkersUF_8Way_1024x683_32u.raw");
const std::string & LabelMarkersOutputFile4 = Path + std::string("PCB_LabelMarkersUF_8Way_1280x720_32u.raw");

const std::string & CompressedMarkerLabelsOutputFile0 = Path + std::string("Lena_CompressedMarkerLabelsUF_8Way_512x512_32u.raw");
const std::string & CompressedMarkerLabelsOutputFile1 = Path + std::string("CT_skull_CompressedMarkerLabelsUF_8Way_512x512_32u.raw");
const std::string & CompressedMarkerLabelsOutputFile2 = Path + std::string("PCB_METAL_CompressedMarkerLabelsUF_8Way_509x335_32u.raw");
const std::string & CompressedMarkerLabelsOutputFile3 = Path + std::string("PCB2_CompressedMarkerLabelsUF_8Way_1024x683_32u.raw");
const std::string & CompressedMarkerLabelsOutputFile4 = Path + std::string("PCB_CompressedMarkerLabelsUF_8Way_1280x720_32u.raw");

const std::string & LabelMarkersBatchOutputFile0 = Path + std::string("Lena_LabelMarkersUFBatch_8Way_512x512_32u.raw");
const std::string & LabelMarkersBatchOutputFile1 = Path + std::string("CT_skull_LabelMarkersUFBatch_8Way_512x512_32u.raw");
const std::string & LabelMarkersBatchOutputFile2 = Path + std::string("PCB_METAL_LabelMarkersUFBatch_8Way_509x335_32u.raw");
const std::string & LabelMarkersBatchOutputFile3 = Path + std::string("PCB2_LabelMarkersUFBatch_8Way_1024x683_32u.raw");
const std::string & LabelMarkersBatchOutputFile4 = Path + std::string("PCB_LabelMarkersUFBatch_8Way_1280x720_32u.raw");

#ifdef CUDA11U1
const std::string & CompressedMarkerLabelsBatchOutputFile0 = Path + std::string("Lena_CompressedMarkerLabelsUFBatch_8Way_512x512_32u.raw");
const std::string & CompressedMarkerLabelsBatchOutputFile1 = Path + std::string("CT_skull_CompressedMarkerLabelsUFBatch_8Way_512x512_32u.raw");
const std::string & CompressedMarkerLabelsBatchOutputFile2 = Path + std::string("PCB_METAL_CompressedMarkerLabelsUFBatch_8Way_509x335_32u.raw");
const std::string & CompressedMarkerLabelsBatchOutputFile3 = Path + std::string("PCB2_CompressedMarkerLabelsUFBatch_8Way_1024x683_32u.raw");
const std::string & CompressedMarkerLabelsBatchOutputFile4 = Path + std::string("PCB_CompressedMarkerLabelsUFBatch_8Way_1280x720_32u.raw");
#endif

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
        if (nWidth != 509 || nHeight != 335) 
            return -1;
        fopen_s(&bmpFile, InputFile2.c_str(), "rb");
    }
    else if (nImage == 3)
    {
        if (nWidth != 1024 || nHeight != 683) 
            return -1;
        fopen_s(&bmpFile, InputFile3.c_str(), "rb");
    }
    else if (nImage == 4)
    {
        if (nWidth != 1280 || nHeight != 720) 
            return -1;
        fopen_s(&bmpFile, InputFile4.c_str(), "rb");
    }
    else
    {
        printf ("Input file load failed.\n");
        return -1;
    }

    if (bmpFile == NULL) 
        return -1;
    nSize = fread(pImage, 1, nWidth * nHeight, bmpFile);
    if (nSize < nWidth * nHeight)
    {
        fclose(bmpFile);        
        return -1;
    }
    fclose(bmpFile);

    printf ("Input file load succeeded.\n");

    return 0;
}

// *****************************************************************************
// main batched image region connected label marker and compression function
// -----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{

    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
    (pidx = findParamIndex(argv, argc, "--help")) != -1) {
        std::cout << "Usage: " << argv[0]
          << "[-b number-of-batch]\n";
        std::cout << "Parameters: " << std::endl;
        std::cout << "\tnumber-of-batch\t:\tUse number of batch to process [default 5]" << std::endl;
        return EXIT_SUCCESS;
    }

    image_labelmarker_params_t params;

    params.numofbatch = 5;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
    params.numofbatch = std::atoi(argv[pidx + 1]);
    }

    int      aGenerateLabelsScratchBufferSize[NUMBER_OF_IMAGES];
    int      aCompressLabelsScratchBufferSize[NUMBER_OF_IMAGES];

    int nCompressedLabelCount = 0;
    cudaError_t cudaError;
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;
    FILE * bmpFile;

    for (int j = 0; j < params.numofbatch; j++)
    {
        pInputImageDev[j] = 0;
        pInputImageHost[j] = 0;
        pUFGenerateLabelsScratchBufferDev[j] = 0;
        pUFCompressedLabelsScratchBufferDev[j] = 0;
        pUFLabelDev[j] = 0;
        pUFLabelHost[j] = 0;
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
            oSizeROI[nImage].width = 509;
            oSizeROI[nImage].height = 335;
        }
        else if (nImage == 3)
        {
            oSizeROI[nImage].width = 1024;
            oSizeROI[nImage].height = 683;
        }
        else if (nImage == 4)
        {
            oSizeROI[nImage].width = 1280;
            oSizeROI[nImage].height = 720;
        }

        // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can significantly improve the performance of many NPP functions, 
        // for UF function label markers generation or compression DO NOT USE cudaMallocPitch().  Doing so could result in incorrect output.

        cudaError = cudaMalloc ((void**)&pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        // For images processed with UF label markers functions ROI width and height for label markers generation output AND marker compression functions MUST be the same AND 
        // line pitch MUST be equal to ROI.width * sizeof(Npp32u).  Also the image pointer used for label markers generation output must start at the same position in the image
        // as it does in the marker compression function.  Also note that actual input image size and ROI do not necessarily need to be related other than ROI being less than
        // or equal to image size and image starting position does not necessarily have to be at pixel 0 in the input image.

        cudaError = cudaMalloc ((void**)&pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pInputImageHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height));
        pUFLabelHost[nImage] = reinterpret_cast<Npp32u *>(malloc(oSizeROI[nImage].width * sizeof(Npp32u) * oSizeROI[nImage].height));

        // Use UF functions throughout this sample.

        nppStatus = nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI[nImage], &aGenerateLabelsScratchBufferSize[nImage]);

        // One at a time image processing

        cudaError = cudaMalloc ((void **)&pUFGenerateLabelsScratchBufferDev[nImage], aGenerateLabelsScratchBufferSize[nImage]);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        if (loadRaw8BitImage(pInputImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, nImage) == 0)
        {
            cudaError = cudaMemcpy2DAsync(pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage], 
                                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, 
                                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            nppStatus = nppPlusV::nppiLabelMarkersUF_8u32u_C1R_Ctx(pInputImageDev[nImage],
                                                         oSizeROI[nImage].width * sizeof(Npp8u),
                                                         pUFLabelDev[nImage],
                                                         oSizeROI[nImage].width * sizeof(Npp32u),
                                                         oSizeROI[nImage],
                                                         nppiNormInf,
                                                         pUFGenerateLabelsScratchBufferDev[nImage],
                                                         nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("Lena_LabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 1)
                    printf("CT_skull_LabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 2)
                    printf("PCB_METAL_LabelMarkersUF_8Way_509x335_32u failed.\n");
                else if (nImage == 3)
                    printf("PCB2_LabelMarkersUF_8Way_1024x683_32u failed.\n");
                else if (nImage == 4)
                    printf("PCB_LabelMarkersUF_8Way_1280x720_32u failed.\n");
                tearDown();
                return -1;
            }

            cudaError = cudaMemcpy2DAsync(pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u), 
                                          pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post label generation cudaStreamSynchronize failed\n");
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
                nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            nppStatus = nppPlusV::nppiCompressMarkerLabelsGetBufferSize_32u_C1R(oSizeROI[nImage].width * oSizeROI[nImage].height, &aCompressLabelsScratchBufferSize[nImage]);
            if (nppStatus != NPP_NO_ERROR)
                return nppStatus;

            cudaError = cudaMalloc ((void **)&pUFCompressedLabelsScratchBufferDev[nImage], aCompressLabelsScratchBufferSize[nImage]);
            if (cudaError != cudaSuccess)
                return NPP_MEMORY_ALLOCATION_ERR;

            nCompressedLabelCount = 0;

            nppStatus = nppPlusV::nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage], 
                                                            oSizeROI[nImage].width * oSizeROI[nImage].height, &nCompressedLabelCount, 
                                                            pUFCompressedLabelsScratchBufferDev[nImage],nppStreamCtx);

            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("Lena_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 1)
                    printf("CT_Skull_CompressedLabelMarkersUF_8Way_512x512_32u failed.\n");
                else if (nImage == 2)
                    printf("PCB_METAL_CompressedLabelMarkersUF_8Way_509x335_32u failed.\n");
                else if (nImage == 3)
                    printf("PCB2_CompressedLabelMarkersUF_8Way_1024x683_32u failed.\n");
                else if (nImage == 4)
                    printf("PCB_CompressedLabelMarkersUF_8Way_1280x720_32u failed.\n");
                tearDown();
                return -1;
            }

            cudaError = cudaMemcpy2DAsync(pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u), 
                                          pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait for host image read backs to finish, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess || nCompressedLabelCount == 0) 
            {
                printf ("Post label compression cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                fopen_s(&bmpFile, CompressedMarkerLabelsOutputFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, CompressedMarkerLabelsOutputFile1.c_str(), "wb");
            else if (nImage == 2)
                fopen_s(&bmpFile, CompressedMarkerLabelsOutputFile2.c_str(), "wb");
            else if (nImage == 3)
                fopen_s(&bmpFile, CompressedMarkerLabelsOutputFile3.c_str(), "wb");
            else if (nImage == 4)
                fopen_s(&bmpFile, CompressedMarkerLabelsOutputFile4.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            if (nImage == 0)
                printf("Lena_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, compressed label count is %d.\n", nCompressedLabelCount);
            else if (nImage == 1)
                printf("CT_Skull_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, compressed label count is %d.\n", nCompressedLabelCount);
            else if (nImage == 2)
                printf("PCB_METAL_CompressedMarkerLabelsUF_8Way_509x335_32u succeeded, compressed label count is %d.\n", nCompressedLabelCount);
            else if (nImage == 3)
                printf("PCB2_CompressedMarkerLabelsUF_8Way_1024x683_32u succeeded, compressed label count is %d.\n", nCompressedLabelCount);
            else if (nImage == 4)
                printf("PCB_CompressedMarkerLabelsUF_8Way_1280x720_32u succeeded, compressed label count is %d.\n", nCompressedLabelCount);
        }
    }

    // Batch image processing

    // We want to allocate scratch buffers more efficiently for batch processing so first we free up the scratch buffers for image 0 and reallocate them.
    // This is not required but helps cudaMalloc to work more efficiently.

    cudaFree(pUFCompressedLabelsScratchBufferDev[0]);

    int nTotalBatchedUFCompressLabelsScratchBufferDevSize = 0;

    for (int k = 0; k < NUMBER_OF_IMAGES; k++)
        nTotalBatchedUFCompressLabelsScratchBufferDevSize += aCompressLabelsScratchBufferSize[k];

    cudaError = cudaMalloc ((void **)&pUFCompressedLabelsScratchBufferDev[0], nTotalBatchedUFCompressLabelsScratchBufferDevSize);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    // Now allocate batch lists

    int nBatchImageListBytes = NUMBER_OF_IMAGES * sizeof(NppiImageDescriptor);

    cudaError = cudaMalloc ((void**)&pUFBatchSrcImageListDev, nBatchImageListBytes);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    cudaError = cudaMalloc ((void**)&pUFBatchSrcDstImageListDev, nBatchImageListBytes);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    pUFBatchSrcImageListHost = reinterpret_cast<NppiImageDescriptor *>(malloc(nBatchImageListBytes));
    pUFBatchSrcDstImageListHost = reinterpret_cast<NppiImageDescriptor *>(malloc(nBatchImageListBytes));

    NppiSize oMaxROISize = {0, 0};

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        pUFBatchSrcImageListHost[nImage].pData = pInputImageDev[nImage];
        pUFBatchSrcImageListHost[nImage].nStep = oSizeROI[nImage].width * sizeof(Npp8u);
        // src image oSize parameter is ignored in these NPP functions
        pUFBatchSrcDstImageListHost[nImage].pData = pUFLabelDev[nImage];
        pUFBatchSrcDstImageListHost[nImage].nStep = oSizeROI[nImage].width * sizeof(Npp32u);
        pUFBatchSrcDstImageListHost[nImage].oSize = oSizeROI[nImage];
        if (oSizeROI[nImage].width > oMaxROISize.width)
            oMaxROISize.width = oSizeROI[nImage].width;
        if (oSizeROI[nImage].height > oMaxROISize.height)
            oMaxROISize.height = oSizeROI[nImage].height;
    }

    // Copy label generation batch lists from CPU to GPU
    cudaError = cudaMemcpyAsync(pUFBatchSrcImageListDev, pUFBatchSrcImageListHost, nBatchImageListBytes, cudaMemcpyHostToDevice, nppStreamCtx.hStream);
    if (cudaError != cudaSuccess)
        return NPP_MEMCPY_ERROR;

    cudaError = cudaMemcpyAsync(pUFBatchSrcDstImageListDev, pUFBatchSrcDstImageListHost, nBatchImageListBytes, cudaMemcpyHostToDevice, nppStreamCtx.hStream);
    if (cudaError != cudaSuccess)
        return NPP_MEMCPY_ERROR;

    // We use 8-way neighbor search throughout this example
    nppStatus = nppPlusV::nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(pUFBatchSrcImageListDev, pUFBatchSrcDstImageListDev, 
                                                               NUMBER_OF_IMAGES, oMaxROISize, nppiNormInf, nppStreamCtx);

    if (nppStatus != NPP_SUCCESS)  
    {
        printf("LabelMarkersUFBatch_8Way_8u32u failed.\n");
        tearDown();
        return -1;
    }

    // Now read back generated device images to the host

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        cudaError = cudaMemcpy2DAsync(pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u), 
                                      pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
    }

    // Wait for host image read backs to complete, not necessary if no need to synchronize
    if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
    {
        printf ("Post label generation cudaStreamSynchronize failed\n");
        tearDown();
        return -1;
    }

    // Save output to files
    for (int nImage = 0; nImage < params.numofbatch; nImage++)
    {
        if (nImage == 0)
            fopen_s(&bmpFile, LabelMarkersBatchOutputFile0.c_str(), "wb");
        else if (nImage == 1)
            fopen_s(&bmpFile, LabelMarkersBatchOutputFile1.c_str(), "wb");
        else if (nImage == 2)
            fopen_s(&bmpFile, LabelMarkersBatchOutputFile2.c_str(), "wb");
        else if (nImage == 3)
            fopen_s(&bmpFile, LabelMarkersBatchOutputFile3.c_str(), "wb");
        else if (nImage == 4)
            fopen_s(&bmpFile, LabelMarkersBatchOutputFile4.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        size_t nSize = 0;
        for (int j = 0; j < oSizeROI[nImage].height; j++)
        {
            nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
        }
        fclose(bmpFile);
    }

#ifdef CUDA11U1

    // Now allocate scratch buffer memory for batched label compression
    cudaError = cudaMalloc ((void**)&pUFBatchSrcDstScratchBufferListDev, NUMBER_OF_IMAGES * sizeof(NppiBufferDescriptor));
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    cudaError = cudaMalloc ((void**)&pUFBatchPerImageCompressedCountListDev, NUMBER_OF_IMAGES * sizeof(Npp32u));
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    // Allocate host side scratch buffer point and size list and initialize with device scratch buffer pointers 
    pUFBatchSrcDstScratchBufferListHost = reinterpret_cast<NppiBufferDescriptor *>(malloc(NUMBER_OF_IMAGES * sizeof(NppiBufferDescriptor)));

    pUFBatchPerImageCompressedCountListHost = reinterpret_cast<Npp32u *>(malloc(NUMBER_OF_IMAGES * sizeof(Npp32u)));

    // Start buffer pointer at beginning of full per image buffer list sized pUFCompressedLabelsScratchBufferDev[0]
    Npp32u * pCurUFCompressedLabelsScratchBufferDev = reinterpret_cast<Npp32u *>(pUFCompressedLabelsScratchBufferDev[0]);

    int nMaxUFCompressedLabelsScratchBufferSize = 0;

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        // This particular function works on in-place data and SrcDst image batch list has already been initialized in batched label generation function setup

        //  Initialize each per image buffer descriptor
        pUFBatchSrcDstScratchBufferListHost[nImage].pData = reinterpret_cast<void *>(pCurUFCompressedLabelsScratchBufferDev);
        pUFBatchSrcDstScratchBufferListHost[nImage].nBufferSize = aCompressLabelsScratchBufferSize[nImage];

        if (aCompressLabelsScratchBufferSize[nImage] > nMaxUFCompressedLabelsScratchBufferSize)
            nMaxUFCompressedLabelsScratchBufferSize = aCompressLabelsScratchBufferSize[nImage];

        // Offset buffer pointer to next per image buffer
        Npp8u * pTempBuffer =  reinterpret_cast<Npp8u *>(pCurUFCompressedLabelsScratchBufferDev);
        pTempBuffer += aCompressLabelsScratchBufferSize[nImage];
        pCurUFCompressedLabelsScratchBufferDev = reinterpret_cast<Npp32u *>((void *)(pTempBuffer));
    }

    // Copy compression batch scratch buffer list from CPU to GPU
    cudaError = cudaMemcpyAsync(pUFBatchSrcDstScratchBufferListDev, pUFBatchSrcDstScratchBufferListHost, NUMBER_OF_IMAGES * sizeof(NppiBufferDescriptor), cudaMemcpyHostToDevice, nppStreamCtx.hStream);
    if (cudaError != cudaSuccess)
        return NPP_MEMCPY_ERROR;

    nppStatus = nppPlusV::nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(pUFBatchSrcDstImageListDev, pUFBatchSrcDstScratchBufferListDev, pUFBatchPerImageCompressedCountListDev,
                                                                      NUMBER_OF_IMAGES, oMaxROISize, nMaxUFCompressedLabelsScratchBufferSize, nppStreamCtx);
    if (nppStatus != NPP_SUCCESS)  
    {
        printf("BatchCompressedLabelMarkersUF_8Way_32u failed.\n");
        tearDown();
        return -1;
    }

    // Copy output compressed label images back to host
    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        cudaError = cudaMemcpy2DAsync(pUFLabelHost[nImage], oSizeROI[nImage].width * sizeof(Npp32u), 
                                      pUFLabelDev[nImage], oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].width * sizeof(Npp32u), oSizeROI[nImage].height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
    }

    // Wait for host image read backs to complete, not necessary if no need to synchronize
    if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
    {
        printf ("Post label compression cudaStreamSynchronize failed\n");
        tearDown();
        return -1;
    }

    // Save compressed label images into files
    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        if (nImage == 0)
            fopen_s(&bmpFile, CompressedMarkerLabelsBatchOutputFile0.c_str(), "wb");
        else if (nImage == 1)
            fopen_s(&bmpFile, CompressedMarkerLabelsBatchOutputFile1.c_str(), "wb");
        else if (nImage == 2)
            fopen_s(&bmpFile, CompressedMarkerLabelsBatchOutputFile2.c_str(), "wb");
        else if (nImage == 3)
            fopen_s(&bmpFile, CompressedMarkerLabelsBatchOutputFile3.c_str(), "wb");
        else if (nImage == 4)
            fopen_s(&bmpFile, CompressedMarkerLabelsBatchOutputFile4.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        size_t nSize = 0;
        for (int j = 0; j < oSizeROI[nImage].height; j++)
        {
            nSize += fwrite(&pUFLabelHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp32u), oSizeROI[nImage].width, bmpFile);
        }
        fclose(bmpFile);
    }

    // Read back per image compressed label count.
    cudaError = cudaMemcpyAsync(pUFBatchPerImageCompressedCountListHost, pUFBatchPerImageCompressedCountListDev, 
                                NUMBER_OF_IMAGES * sizeof(Npp32u), cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
    if (cudaError != cudaSuccess)
    {
        tearDown();
        return NPP_MEMCPY_ERROR;
    }

    // Wait for host read back to complete
    cudaError = cudaStreamSynchronize(nppStreamCtx.hStream);

    printf("\n\n");

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        if (nImage == 0)
            printf("Lena_CompressedMarkerLabelsUFBatch_8Way_512x512_32u succeeded, compressed label count is %d.\n", pUFBatchPerImageCompressedCountListHost[nImage]);
        else if (nImage == 1)
            printf("CT_Skull_CompressedMarkerLabelsUFBatch_8Way_512x512_32u succeeded, compressed label count is %d.\n", pUFBatchPerImageCompressedCountListHost[nImage]);
        else if (nImage == 2)
            printf("PCB_METAL_CompressedMarkerLabelsUFBatch_8Way_509x335_32u succeeded, compressed label count is %d.\n", pUFBatchPerImageCompressedCountListHost[nImage]);
        else if (nImage == 3)
            printf("PCB2_CompressedMarkerLabelsUFBatch_8Way_1024x683_32u succeeded, compressed label count is %d.\n", pUFBatchPerImageCompressedCountListHost[nImage]);
        else if (nImage == 4)
            printf("PCB_CompressedMarkerLabelsUFBatch_8Way_1280x720_32u succeeded, compressed label count is %d.\n", pUFBatchPerImageCompressedCountListHost[nImage]);
    }

#endif // CUDA11U1

    tearDown();

    return 0;
}



