/* Copyright 2021 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
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

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <stdio.h>
#include <string.h>
#include <fstream>

#include <npp.h>

// Remove this if compiling on a pre-NPP 11.5 release
#define USE_NPP_11_5

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
//

    Npp8u  * pInputImageDev = 0;
    Npp8u  * pInputImageHost = 0;
    Npp8u  * pUFGenerateLabelsScratchBufferDev = 0;
    Npp8u  * pUFCompressedLabelsScratchBufferDev = 0;
    Npp32u * pUFLabelDev = 0;
    Npp32u * pUFLabelHost = 0;
    NppiCompressedMarkerLabelsInfo * pMarkerLabelsInfoListDev = 0;
    NppiCompressedMarkerLabelsInfo * pMarkerLabelsInfoListHost = 0;
    Npp8u * pContoursImageDev = 0;
    Npp8u * pContoursImageHost = 0;
    NppiContourPixelDirectionInfo * pContoursDirectionImageDev = 0;
    NppiContourPixelDirectionInfo * pContoursDirectionImageHost = 0;
    Npp32u * pContoursPixelCountsListDev = 0;
    Npp32u * pContoursPixelCountsListHost = 0;
    Npp32u * pContoursPixelsFoundListDev = 0;
    Npp32u * pContoursPixelsFoundListHost = 0;
    Npp32u * pContoursPixelStartingOffsetDev = 0;
    Npp32u * pContoursPixelStartingOffsetHost = 0;
    Npp8u * pContoursGeometryImageHost = 0;
    Npp8u * pContoursOrderedGeometryImageHost = 0;

#ifdef USE_NPP_11_5
    NppiContourBlockSegment * pContoursBlockSegmentListDev = 0;
    NppiContourBlockSegment * pContoursBlockSegmentListHost = 0;
#endif

    NppiContourTotalsInfo oContoursTotalsInfoHost;

void tearDown() // Clean up and tear down
{
#ifdef USE_NPP_11_5
    if (pContoursBlockSegmentListDev != 0)
        cudaFree(pContoursBlockSegmentListDev);
#endif
    if (pUFCompressedLabelsScratchBufferDev != 0)
        cudaFree(pUFCompressedLabelsScratchBufferDev);
    if (pUFGenerateLabelsScratchBufferDev != 0)
        cudaFree(pUFGenerateLabelsScratchBufferDev);
    if (pUFLabelDev != 0)
        cudaFree(pUFLabelDev);
    if (pInputImageDev != 0)
        cudaFree(pInputImageDev);
    if (pContoursPixelStartingOffsetDev != 0)
        cudaFree(pContoursPixelStartingOffsetDev); 
    if (pContoursPixelCountsListDev != 0)
        cudaFree(pContoursPixelCountsListDev);
    if (pContoursPixelsFoundListDev != 0)
        cudaFree(pContoursPixelsFoundListDev);
    if (pMarkerLabelsInfoListDev != 0)
        cudaFree(pMarkerLabelsInfoListDev);
    if (pContoursImageDev != 0)
        cudaFree(pContoursImageDev);
    if (pContoursDirectionImageDev != 0)
        cudaFree(pContoursDirectionImageDev);

    if (pUFLabelHost != 0)
        free(pUFLabelHost);
    if (pMarkerLabelsInfoListHost != 0)
        free(pMarkerLabelsInfoListHost);
    if (pContoursImageHost != 0)
        free(pContoursImageHost);
    if (pContoursDirectionImageHost != 0)
        free(pContoursDirectionImageHost);
    if (pContoursPixelCountsListHost != 0)
        free(pContoursPixelCountsListHost);
    if (pContoursPixelsFoundListHost != 0)
        free(pContoursPixelsFoundListHost);
    if (pContoursPixelStartingOffsetHost != 0)
        free(pContoursPixelStartingOffsetHost);
#ifdef USE_NPP_11_5
    if (pContoursBlockSegmentListHost != 0)
        free(pContoursBlockSegmentListHost); 
#endif
    if (pContoursGeometryImageHost != 0)
        free(pContoursGeometryImageHost);
    if (pContoursOrderedGeometryImageHost != 0)
        free(pContoursOrderedGeometryImageHost); 
}

const std::string & Path = std::string("images/");

const std::string & InputFile0 = Path + std::string("CircuitBoard_2048x1024_8u.raw");

const std::string & LabelMarkersOutputFile0 = Path + std::string("CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u.raw");

const std::string & CompressedMarkerLabelsOutputFile0 = Path + std::string("CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u.raw");

const std::string & ContoursOutputFile0 = Path + std::string("CircuitBoard_Contours_8Way_2048x1024_8u.raw");

const std::string & ContoursDirectionOutputFile0 = Path + std::string("CircuitBoard_ContoursDirection_8Way_2048x1024_8u.raw");

const std::string & ContoursReconstructedFile0 = Path + std::string("CircuitBoard_ContoursReconstructed_8Way_2048x1024_8u.raw");

int 
loadRaw8BitImage(Npp8u * pImage, int nWidth, int nHeight, int nImage)
{
    FILE * bmpFile;
    size_t nSize;

    if (nImage == 0)
	{
        if (nWidth != 2048 || nHeight != 1024) 
            return -1;
        bmpFile = fopen(InputFile0.c_str(), "rb");
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

int 
main( int argc, char** argv )
{

    int      aGenerateLabelsScratchBufferSize;
    int      aCompressLabelsScratchBufferSize;

    int nCompressedLabelCount = 0;
    cudaError_t cudaError;
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;
	FILE * bmpFile;

    pInputImageDev = 0;
    pInputImageHost = 0;
    pUFGenerateLabelsScratchBufferDev = 0;
    pUFCompressedLabelsScratchBufferDev = 0;
    pUFLabelDev = 0;
    pUFLabelHost = 0;
    pMarkerLabelsInfoListDev = 0;
    pMarkerLabelsInfoListHost = 0;
    pContoursImageDev = 0;
    pContoursImageHost = 0;
    pContoursDirectionImageDev = 0;
    pContoursDirectionImageHost = 0;
    pContoursPixelCountsListDev = 0;
    pContoursPixelCountsListHost = 0;
    pContoursPixelsFoundListHost = 0;
    pContoursPixelStartingOffsetHost = 0;
    pContoursGeometryImageHost = 0;
#ifdef USE_NPP_11_5
    pContoursBlockSegmentListDev = 0;
    pContoursBlockSegmentListHost = 0;
#endif

    nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

    cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
    {
        printf("CUDA error: no devices supporting CUDA.\n");
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
    }

    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

//    if (libVer->major > 11 || (libVer->major == 11 && libVer->minor >= 5))
//        bUse115 = 1;

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

    NppiSize oSizeROI;

    oSizeROI.width = 2048;
    oSizeROI.height = 1024;

    // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can significantly improve the performance of many NPP functions, 
    // for UF function label markers generation or compression DO NOT USE cudaMallocPitch().  Doing so could result in incorrect output.

    cudaError = cudaMalloc ((void**)&pInputImageDev, oSizeROI.width * sizeof(Npp8u) * oSizeROI.height);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    // For images processed with UF label markers functions ROI width and height for label markers generation output AND marker compression functions MUST be the same AND 
    // line pitch MUST be equal to ROI.width * sizeof(Npp32u).  Also the image pointer used for label markers generation output must start at the same position in the image
    // as it does in the marker compression function.  Also note that actual input image size and ROI do not necessarily need to be related other than ROI being less than
    // or equal to image size and image starting position does not necessarily have to be at pixel 0 in the input image.

    cudaError = cudaMalloc ((void**)&pUFLabelDev, oSizeROI.width * sizeof(Npp32u) * oSizeROI.height);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    pInputImageHost = reinterpret_cast<Npp8u *>(malloc(oSizeROI.width * sizeof(Npp8u) * oSizeROI.height));
    pUFLabelHost = reinterpret_cast<Npp32u *>(malloc(oSizeROI.width * sizeof(Npp32u) * oSizeROI.height));

    // Use UF functions throughout this sample.

    nppStatus = nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI, &aGenerateLabelsScratchBufferSize);

    // One at a time image processing

    cudaError = cudaMalloc ((void **)&pUFGenerateLabelsScratchBufferDev, aGenerateLabelsScratchBufferSize);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    if (loadRaw8BitImage(pInputImageHost, oSizeROI.width * sizeof(Npp8u), oSizeROI.height, 0) == 0)
    {
        cudaError = cudaMemcpy2DAsync(pInputImageDev, oSizeROI.width * sizeof(Npp8u), pInputImageHost, 
                                                      oSizeROI.width * sizeof(Npp8u), oSizeROI.width * sizeof(Npp8u), oSizeROI.height, 
                                                      cudaMemcpyHostToDevice, nppStreamCtx.hStream);

        nppStatus = nppiLabelMarkersUF_8u32u_C1R_Ctx(pInputImageDev,
                                                     oSizeROI.width * sizeof(Npp8u),
                                                     pUFLabelDev,
                                                     oSizeROI.width * sizeof(Npp32u),
                                                     oSizeROI,
                                                     nppiNormInf,
                                                     pUFGenerateLabelsScratchBufferDev,
                                                     nppStreamCtx);

        if (nppStatus != NPP_SUCCESS)  
        {
            printf("CircuitBoard_LabelMarkersUF_8Way_2048x1024_32u failed.\n");
            tearDown();
            return -1;
        }

        cudaError = cudaMemcpy2DAsync(pUFLabelHost, oSizeROI.width * sizeof(Npp32u), 
                                      pUFLabelDev, oSizeROI.width * sizeof(Npp32u), oSizeROI.width * sizeof(Npp32u), oSizeROI.height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

        // Wait host image read backs to complete, not necessary if no need to synchronize
        if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
        {
            printf ("Post label generation cudaStreamSynchronize failed\n");
            tearDown();
            return -1;
        }

        bmpFile = fopen(LabelMarkersOutputFile0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        size_t nSize = 0;
        for (int j = 0; j < oSizeROI.height; j++)
        {
            nSize += fwrite(&pUFLabelHost[j * oSizeROI.width], sizeof(Npp32u), oSizeROI.width, bmpFile);
        }
        fclose(bmpFile);

        nppStatus = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(oSizeROI.width * oSizeROI.height, &aCompressLabelsScratchBufferSize);
        if (nppStatus != NPP_NO_ERROR)
            return nppStatus;

        cudaError = cudaMalloc ((void **)&pUFCompressedLabelsScratchBufferDev, aCompressLabelsScratchBufferSize);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        nCompressedLabelCount = 0;

        nppStatus = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(pUFLabelDev, oSizeROI.width * sizeof(Npp32u), oSizeROI, 
                                                            oSizeROI.width * oSizeROI.height, &nCompressedLabelCount, 
                                                            pUFCompressedLabelsScratchBufferDev, nppStreamCtx);

        if (nppStatus != NPP_SUCCESS)  
        {
            printf("CircuitBoard_CompressedLabelMarkersUF_8Way_2048x1024_32u failed.\n");
            tearDown();
            return -1;
        }

        cudaError = cudaMemcpy2DAsync(pUFLabelHost, oSizeROI.width * sizeof(Npp32u), 
                                      pUFLabelDev, oSizeROI.width * sizeof(Npp32u), oSizeROI.width * sizeof(Npp32u), oSizeROI.height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

        // Wait for host image read backs to finish, not necessary if no need to synchronize
        if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess || nCompressedLabelCount == 0) 
        {
            printf ("Post label compression cudaStreamSynchronize failed\n");
            tearDown();
            return -1;
        }

        bmpFile = fopen(CompressedMarkerLabelsOutputFile0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI.height; j++)
        {
            nSize += fwrite(&pUFLabelHost[j * oSizeROI.width], sizeof(Npp32u), oSizeROI.width, bmpFile);
        }
        fclose(bmpFile);

        printf("CircuitBoard_CompressedMarkerLabelsUF_8Way_2048x1024_32u succeeded, compressed label count is %d.\n", nCompressedLabelCount);

        unsigned int nInfoListSize;

        nppStatus = nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(nCompressedLabelCount, &nInfoListSize);
        if (nppStatus != NPP_NO_ERROR)
            return nppStatus;

        cudaError = cudaMalloc ((void **)&pMarkerLabelsInfoListDev, nInfoListSize);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void **)&pContoursImageDev, oSizeROI.width * sizeof(Npp8u) * oSizeROI.height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void **)&pContoursDirectionImageDev, oSizeROI.width * sizeof(NppiContourPixelDirectionInfo) * oSizeROI.height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void **)&pContoursPixelCountsListDev, sizeof(Npp32u) * (nCompressedLabelCount + 4));
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void **)&pContoursPixelsFoundListDev, sizeof(Npp32u) * (nCompressedLabelCount + 4));
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void **)&pContoursPixelStartingOffsetDev, sizeof(Npp32u) * (nCompressedLabelCount + 4));
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pMarkerLabelsInfoListHost = reinterpret_cast<NppiCompressedMarkerLabelsInfo *>(malloc(nInfoListSize));
        pContoursImageHost = reinterpret_cast<Npp8u *>(malloc(oSizeROI.width * sizeof(Npp8u) * oSizeROI.height));
        pContoursDirectionImageHost = reinterpret_cast<NppiContourPixelDirectionInfo *>(malloc(oSizeROI.width * sizeof(NppiContourPixelDirectionInfo) * oSizeROI.height));
        pContoursPixelCountsListHost = reinterpret_cast<Npp32u *>(malloc(sizeof(Npp32u) * (nCompressedLabelCount + 4)));
        pContoursPixelsFoundListHost = reinterpret_cast<Npp32u *>(malloc(sizeof(Npp32u) * (nCompressedLabelCount + 4)));
        pContoursPixelStartingOffsetHost = reinterpret_cast<Npp32u *>(malloc(sizeof(Npp32u) * (nCompressedLabelCount + 4)));

#ifdef USE_NPP_11_5
        nppStatus = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(pUFLabelDev, 
                                                                 oSizeROI.width * sizeof(Npp32u), 
                                                                 oSizeROI,
                                                                 nCompressedLabelCount, 
                                                                 pMarkerLabelsInfoListDev,
                                                                 pContoursImageDev, 
                                                                 oSizeROI.width * sizeof(Npp8u),
                                                                 pContoursDirectionImageDev, 
                                                                 oSizeROI.width * sizeof(NppiContourPixelDirectionInfo),
                                                                 &oContoursTotalsInfoHost, 
                                                                 pContoursPixelCountsListDev,
                                                                 pContoursPixelCountsListHost, 
                                                                 pContoursPixelStartingOffsetDev, 
                                                                 pContoursPixelStartingOffsetHost, 
                                                                 nppStreamCtx);
#else
        nppStatus = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(pUFLabelDev, 
                                                                 oSizeROI.width * sizeof(Npp32u), 
                                                                 oSizeROI,
                                                                 nCompressedLabelCount, 
                                                                 pMarkerLabelsInfoListDev,
                                                                 pContoursImageDev, 
                                                                 oSizeROI.width * sizeof(Npp8u),
                                                                 pContoursDirectionImageDev, 
                                                                 oSizeROI.width * sizeof(NppiContourPixelDirectionInfo),
                                                                 &oContoursTotalsInfoHost, 
                                                                 pContoursPixelCountsListDev,
                                                                 pContoursPixelCountsListHost, 
                                                                 NULL,
                                                                 pContoursPixelStartingOffsetHost, 
                                                                 nppStreamCtx);
#endif
        cudaError = cudaMemcpy2DAsync(pContoursImageHost, oSizeROI.width * sizeof(Npp8u), 
                                      pContoursImageDev, oSizeROI.width * sizeof(Npp8u), oSizeROI.width * sizeof(Npp8u), oSizeROI.height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

        cudaError = cudaMemcpy2DAsync(pContoursDirectionImageHost, oSizeROI.width * sizeof(NppiContourPixelDirectionInfo), 
                                      pContoursDirectionImageDev, oSizeROI.width * sizeof(NppiContourPixelDirectionInfo), oSizeROI.width * sizeof(NppiContourPixelDirectionInfo), oSizeROI.height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

        // Wait for host image read backs to finish, not necessary if no need to synchronize
        if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess || nCompressedLabelCount == 0) 
        {
            printf ("Post info list cudaStreamSynchronize failed\n");
            tearDown();
            return -1;
        }

        bmpFile = fopen(ContoursOutputFile0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI.height; j++)
        {
            nSize += fwrite(&pContoursImageHost[j * oSizeROI.width], sizeof(Npp8u), oSizeROI.width, bmpFile);
        }
        fclose(bmpFile);

        unsigned int nStartID = 1;
        unsigned int nStopID = nCompressedLabelCount;

#ifdef USE_NPP_11_5
        unsigned int nContoursBlockSegmentListSize;

        nppStatus = nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(pContoursPixelCountsListHost,
                                                                                    oContoursTotalsInfoHost.nTotalImagePixelContourCount, 
                                                                                    nCompressedLabelCount,
                                                                                    nStartID,
                                                                                    nStopID,
                                                                                    &nContoursBlockSegmentListSize);

        cudaError = cudaMalloc ((void **)&pContoursBlockSegmentListDev, nContoursBlockSegmentListSize);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pContoursBlockSegmentListHost = reinterpret_cast<NppiContourBlockSegment *>(malloc(nContoursBlockSegmentListSize));
#endif

        unsigned int nGeometryListSize;

        nppStatus = nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(pContoursPixelStartingOffsetHost[nCompressedLabelCount],
                                                                         &nGeometryListSize);

        NppiContourPixelGeometryInfo * pContoursPixelGeometryListsHost = reinterpret_cast<NppiContourPixelGeometryInfo *>(malloc(nGeometryListSize));
        NppiContourPixelGeometryInfo * pContoursPixelGeometryListsDev;

        cudaError = cudaMalloc ((void **)&pContoursPixelGeometryListsDev, nGeometryListSize);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaStreamSynchronize(nppStreamCtx.hStream);

        cudaError = cudaMemcpyAsync(pContoursDirectionImageHost, pContoursDirectionImageDev, oSizeROI.width * sizeof(NppiContourPixelDirectionInfo) * oSizeROI.height,
                                    cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

        pContoursGeometryImageHost = reinterpret_cast<Npp8u *>(malloc(oSizeROI.width * sizeof(Npp8u) * oSizeROI.height));
        pContoursOrderedGeometryImageHost = reinterpret_cast<Npp8u *>(malloc(oSizeROI.width * sizeof(Npp8u) * oSizeROI.height));

        if (pContoursOrderedGeometryImageHost != 0)
            memset(pContoursOrderedGeometryImageHost, 255, oSizeROI.width * sizeof(Npp8u) * oSizeROI.height);

        cudaStreamSynchronize(nppStreamCtx.hStream);

/*
 * Note that to significantly improve performance by default a contour that contains more than 256K pixels will be bypassed when generating the 
 * output geometry list. The contour ID and number of contour pixels will be output in the contour list however. You can still get this function 
 * to output the geometry of this size contour however by calling the function with a starting contour ID of that contour ID and ending contour 
 * ID of that contour ID + 1.  Note that doing so for contours approaching a million pixels can take many minutes. Also, due to the structure of 
 * some images contour ID 0 can contain ALL contours in the image so setting the starting contour ID to 1 can significantly increase output 
 * preprocessing performance. 
 *  
 * Once nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx() has been called with a particular 
 * range of contour IDs nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R() can be recalled any number of times 
 * with any range of contour IDs that were included in the preceeding corresponding 
 * nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx() call. 
 *  
 * In general a few contour pixels can escape insertion into the ordered geometry list.  When geometry output is in clockwise order the 
 * extra pixels will be at the end of the geometry list otherwise they will be at the start of the list.
 *  
 */
#ifdef USE_NPP_11_5
        nppStatus = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(pMarkerLabelsInfoListDev,
                                                                                      pMarkerLabelsInfoListHost,
                                                                                      pContoursDirectionImageDev, 
                                                                                      oSizeROI.width * sizeof(NppiContourPixelDirectionInfo),
                                                                                      pContoursPixelGeometryListsDev,
                                                                                      pContoursPixelGeometryListsHost,
                                                                                      pContoursGeometryImageHost,
                                                                                      oSizeROI.width * sizeof(Npp8u),
                                                                                      pContoursPixelCountsListDev,
                                                                                      pContoursPixelsFoundListDev,
                                                                                      pContoursPixelsFoundListHost,
                                                                                      pContoursPixelStartingOffsetDev,
                                                                                      pContoursPixelStartingOffsetHost,
                                                                                      oContoursTotalsInfoHost.nTotalImagePixelContourCount,
                                                                                      nCompressedLabelCount,
                                                                                      nStartID,
                                                                                      nStopID,
                                                                                      pContoursBlockSegmentListDev,
                                                                                      pContoursBlockSegmentListHost,
                                                                                      1, // Counterclockwise contoure geometry list output
                                                                                      oSizeROI,
                                                                                      nppStreamCtx);

        if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess || nCompressedLabelCount == 0) 
        {
            printf ("Post geometry list generation cudaStreamSynchronize failed\n");
            tearDown();
            return -1;
        }
#else
        nppStatus = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(pMarkerLabelsInfoListDev,
                                                                                      pMarkerLabelsInfoListHost, 
                                                                                      pContoursDirectionImageHost, oSizeROI.width * sizeof(NppiContourPixelDirectionInfo),
                                                                                      pContoursPixelGeometryListsHost,
                                                                                      pContoursPixelCountsListHost,
                                                                                      pContoursPixelsFoundListHost,
                                                                                      pContoursPixelStartingOffsetHost,
                                                                                      nCompressedLabelCount,
                                                                                      nStartID,
                                                                                      nStopID,
                                                                                      oSizeROI,
                                                                                      nppStreamCtx);

        if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess || nCompressedLabelCount == 0) 
        {
            printf ("Post geometry list generation cudaStreamSynchronize failed\n");
            tearDown();
            return -1;
        }

        nppStatus = nppiCompressedMarkerLabelsUFContoursOutputGeometryLists_C1R(pContoursPixelGeometryListsHost,
                                                                                pMarkerLabelsInfoListHost, 
                                                                                pContoursGeometryImageHost,
                                                                                oSizeROI.width * sizeof(Npp8u),
                                                                                pContoursPixelCountsListHost,
                                                                                pContoursPixelsFoundListHost,
                                                                                pContoursPixelStartingOffsetHost,
                                                                                nCompressedLabelCount,
                                                                                nStartID,
                                                                                nStopID,
                                                                                1, // Counterclockwise contoure geometry list output
                                                                                oSizeROI);

#endif
        unsigned int nContourPixelCountLimit = 262144;

        for (unsigned int nID = nStartID; nID < nStopID; nID++)
        {
            NppiContourPixelGeometryInfo * pCurContoursPixelGeometryListHost = &pContoursPixelGeometryListsHost[pContoursPixelStartingOffsetHost[nID]]; 
            unsigned int nMaxContourPixelCount = pContoursPixelsFoundListHost[nID];
            int nCurPixelX;
            int nCurPixelY;
            unsigned int nContourPixelCount = 0;
            unsigned char nGrayLevel;

            printf("nID %d Cnt %d BB %d %d %d %d  \n", nID, 
                                                       nMaxContourPixelCount, 
                                                       pCurContoursPixelGeometryListHost[0].oContourPrevPixelLocation.x, 
                                                       pCurContoursPixelGeometryListHost[0].oContourPrevPixelLocation.y, 
                                                       pCurContoursPixelGeometryListHost[0].oContourNextPixelLocation.x, 
                                                       pCurContoursPixelGeometryListHost[0].oContourNextPixelLocation.y);

            unsigned int bOKToOutput = 0;
            if (nMaxContourPixelCount < nContourPixelCountLimit)
                bOKToOutput = 1;
            if (nStopID - nStartID <= 1)
                bOKToOutput = 1;

            if (bOKToOutput)
            {
                nGrayLevel = 240;
                while (nContourPixelCount < static_cast<int>(nMaxContourPixelCount - 1))
                {
                    nCurPixelX = pCurContoursPixelGeometryListHost[nContourPixelCount].oContourOrderedGeometryLocation.x;
                    nCurPixelY = pCurContoursPixelGeometryListHost[nContourPixelCount].oContourOrderedGeometryLocation.y;
//                    if (nCurPixelX >= 0)
                        printf("ID %d Cnt %d %d %d  \n", nID, nContourPixelCount, nCurPixelX, nCurPixelY);
                    if (nCurPixelX >= 0 && nCurPixelY >= 0)
                    {
                        pContoursOrderedGeometryImageHost[oSizeROI.width * nCurPixelY + nCurPixelX] = nGrayLevel;
                        if (nGrayLevel > 0)
                            nGrayLevel -= 1;
                        else
                            nGrayLevel = 240;
                    }
                    nContourPixelCount += 1;
                }
            }
        }

#if 0
        bmpFile = fopen(ContoursDirectionOutputFile0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI.height; j++)
        {
            nSize += fwrite(&pContoursDirectionImageHost[j * oSizeROI.width], sizeof(NppiContourPixelDirectionInfo), oSizeROI.width, bmpFile);
        }
        fclose(bmpFile);
#endif

        bmpFile = fopen(ContoursReconstructedFile0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI.height; j++)
        {
            nSize += fwrite(&pContoursGeometryImageHost[j * oSizeROI.width], sizeof(Npp8u), oSizeROI.width, bmpFile);
        }
        fclose(bmpFile);
    }

    tearDown();

    return 0;
}

