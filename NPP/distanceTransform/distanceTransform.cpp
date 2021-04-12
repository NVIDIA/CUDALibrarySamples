/* Copyright 2021 NVIDIA Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>

#include <nppdefs.h>
#include <nppcore.h>
#include <nppi_filtering_functions.h>

// Set path to whereever your source image files are located.
// In this sample output files are output at the same path location.
const std::string & Path = std::string("../images/");

// This sample uses raw format input and output images so that no image decoder/encoder is necessary.
const std::string & InputFile0 = Path + std::string("Dolphin1_313x317_8u.raw");
const std::string & InputFile1 = Path + std::string("TestImage3_diamond_64x64_8u.raw");

const std::string & OutputVoronoiFile0 = Path + std::string("DistanceTransformVoronoi_Dolphin1_626x317_16s.raw");
const std::string & OutputVoronoiFile1 = Path + std::string("DistanceTransformVoronoi_TestImage3_128x64_16s.raw");

const std::string & OutputTrueTransformFile0 = Path + std::string("DistanceTransformTrue_Dolphin1_313x317_32f.raw");
const std::string & OutputTrueTransformFile1 = Path + std::string("DistanceTransformTrue_TestImage3_diamond_64x64_32f.raw");

const std::string & OutputTruncatedTransformFile0 = Path + std::string("DistanceTransformTruncated_Dolphin1_313x317_16u.raw");
const std::string & OutputTruncatedTransformFile1 = Path + std::string("DistanceTransformTruncated_TestImage3_diamond_64x64_16u.raw");

Npp8u * pInputImage0_8u_Host = 0;
Npp8u * pInputImage1_8u_Host = 0;

Npp8u * pInputImage0_8u_Device = 0;
Npp8u * pInputImage1_8u_Device = 0;

Npp16s * pOutputVoronoiDiagram0_16s_Host = 0;
Npp16s * pOutputVoronoiDiagram1_16s_Host = 0;

Npp16s * pOutputVoronoiDiagram0_16s_Device = 0;
Npp16s * pOutputVoronoiDiagram1_16s_Device = 0;

Npp16u * pOutputTruncatedImage0_16u_Host = 0;
Npp16u * pOutputTruncatedImage1_16u_Host = 0;

Npp16u * pOutputTruncatedImage0_16u_Device = 0;
Npp16u * pOutputTruncatedImage1_16u_Device = 0;

Npp32f * pOutputTransformImage0_32f_Host = 0;
Npp32f * pOutputTransformImage1_32f_Host = 0;

Npp32f * pOutputTransformImage0_32f_Device = 0;
Npp32f * pOutputTransformImage1_32f_Device = 0;

Npp8u * pScratchDeviceBuffer[6] = {0, 0, 0, 0, 0, 0};

int nImage0Width = 313;
int nImage0Height = 317;
int nImage1Width = 64;
int nImage1Height = 64;

// Image ROI is the same as image size in this sample.
NppiSize oImageSizeROI[2];

int 
loadRaw8BitImage(Npp8u * pImage, int nImage)
{
    FILE * rawInputFile;
    size_t nSize;

    if (pImage == 0)
        return -1;

    if (nImage == 0)
    {
        oImageSizeROI[nImage].width = nImage0Width;
        oImageSizeROI[nImage].height = nImage0Height;
        rawInputFile = fopen(InputFile0.c_str(), "rb");
    }
    else if (nImage == 1)
    {
        oImageSizeROI[nImage].width = nImage1Width;
        oImageSizeROI[nImage].height = nImage1Height;
        rawInputFile = fopen(InputFile1.c_str(), "rb");
    }
    else
    {
        printf ("Input file load failed.\n");
        return -1;
    }

    if (rawInputFile == NULL) 
        return -1;
    nSize = fread(pImage, 1, oImageSizeROI[nImage].width * oImageSizeROI[nImage].height, rawInputFile);
    if (nSize < oImageSizeROI[nImage].width * oImageSizeROI[nImage].height)
    {
        fclose(rawInputFile);        
        return -1;
    }
    fclose(rawInputFile);

    printf ("Input file load succeeded.\n");

    return 0;
}

// Deallocate all allocated memory
void shutDown()
{
    for (int i = 0; i < 6; i++)
    {
        if (pScratchDeviceBuffer[i] != 0)
        {
            cudaFree(pScratchDeviceBuffer[i]);
            pScratchDeviceBuffer[i] = 0;
        }
    }

    if (pOutputTransformImage0_32f_Device != 0)
    {
        cudaFree(pOutputTransformImage0_32f_Device);
        pOutputTransformImage0_32f_Device = 0;
    }

    if (pOutputTransformImage1_32f_Device != 0)
    {
        cudaFree(pOutputTransformImage1_32f_Device);
        pOutputTransformImage1_32f_Device = 0;
    }

    if (pOutputTruncatedImage0_16u_Device != 0)
    {
        cudaFree(pOutputTruncatedImage0_16u_Device);
        pOutputTruncatedImage0_16u_Device = 0;
    }

    if (pOutputTruncatedImage1_16u_Device != 0)
    {
        cudaFree(pOutputTruncatedImage1_16u_Device);
        pOutputTruncatedImage1_16u_Device = 0;
    }

    if (pOutputVoronoiDiagram0_16s_Device != 0)
    {
        cudaFree(pOutputVoronoiDiagram0_16s_Device);
        pOutputVoronoiDiagram0_16s_Device = 0;
    }

    if (pOutputVoronoiDiagram1_16s_Device != 0)
    {
        cudaFree(pOutputVoronoiDiagram1_16s_Device);
        pOutputVoronoiDiagram1_16s_Device = 0;
    }

    if (pInputImage0_8u_Device != 0)
    {
        cudaFree(pInputImage0_8u_Device);
        pInputImage0_8u_Device = 0;
    }

    if (pInputImage1_8u_Device != 0)
    {
        cudaFree(pInputImage1_8u_Device);
        pInputImage1_8u_Device = 0;
    }


    if (pOutputTransformImage0_32f_Host != 0)
    {
        free(pOutputTransformImage0_32f_Host);
        pOutputTransformImage0_32f_Host = 0;
    }

    if (pOutputTransformImage1_32f_Host != 0)
    {
        free(pOutputTransformImage1_32f_Host);
        pOutputTransformImage1_32f_Host = 0;
    }

    if (pOutputTruncatedImage0_16u_Host != 0)
    {
        free(pOutputTruncatedImage0_16u_Host);
        pOutputTruncatedImage0_16u_Host = 0;
    }

    if (pOutputTruncatedImage1_16u_Host != 0)
    {
        free(pOutputTruncatedImage1_16u_Host);
        pOutputTruncatedImage1_16u_Host = 0;
    }

    if (pOutputVoronoiDiagram0_16s_Host != 0)
    {
        free(pOutputVoronoiDiagram0_16s_Host);
        pOutputVoronoiDiagram0_16s_Host = 0;
    }

    if (pOutputVoronoiDiagram1_16s_Host != 0)
    {
        free(pOutputVoronoiDiagram1_16s_Host);
        pOutputVoronoiDiagram1_16s_Host = 0;
    }

    if (pInputImage0_8u_Host != 0)
    {
        free(pInputImage0_8u_Host);
        pInputImage0_8u_Host = 0;
    }

    if (pInputImage1_8u_Host != 0)
    {
        free(pInputImage1_8u_Host);
        pInputImage1_8u_Host = 0;
    }
}

// Since NPP is a very large library you should statically link to it whenever possible and only to the NPP libraries that
// contain functions that you use along with nppcore (nppc).  This will eliminate what can be significant library load times when using dynamic libraries.

int main(int argc,char **argv)
{
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;

    nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your created stream ID is if not the NULL stream.

    cudaError_t cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
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

    pInputImage0_8u_Host = reinterpret_cast<Npp8u *>(malloc(nImage0Width * sizeof(Npp8u) * nImage0Height));
    pInputImage1_8u_Host = reinterpret_cast<Npp8u *>(malloc(nImage1Width * sizeof(Npp8u) * nImage1Height));

    if (loadRaw8BitImage(pInputImage0_8u_Host, 0) != 0)
    {
        shutDown();
        return -1;
    }

    if (loadRaw8BitImage(pInputImage1_8u_Host, 1) != 0)
    {
        shutDown();
        return -1;
    }

    pOutputTransformImage0_32f_Host = reinterpret_cast<Npp32f *>(malloc(nImage0Width * sizeof(Npp32f) * nImage0Height));
    pOutputTransformImage1_32f_Host = reinterpret_cast<Npp32f *>(malloc(nImage1Width * sizeof(Npp32f) * nImage1Height));

    pOutputTruncatedImage0_16u_Host = reinterpret_cast<Npp16u *>(malloc(nImage0Width * sizeof(Npp16u) * nImage0Height));
    pOutputTruncatedImage1_16u_Host = reinterpret_cast<Npp16u *>(malloc(nImage1Width * sizeof(Npp16u) * nImage1Height));

    pOutputVoronoiDiagram0_16s_Host = reinterpret_cast<Npp16s *>(malloc(nImage0Width * 2 * sizeof(Npp16s) * nImage0Height));
    pOutputVoronoiDiagram1_16s_Host = reinterpret_cast<Npp16s *>(malloc(nImage1Width * 2 * sizeof(Npp16s) * nImage1Height));

    if (pOutputTransformImage0_32f_Host == 0 || pOutputTransformImage1_32f_Host == 0 ||
        pOutputTruncatedImage0_16u_Host == 0 || pOutputTruncatedImage1_16u_Host == 0 ||
        pOutputVoronoiDiagram0_16s_Host == 0 || pOutputVoronoiDiagram1_16s_Host == 0)
    {
        shutDown();
        return -1;
    }

    size_t nScratchBufferSize;

    for (int i = 0; i < 2; i++)
    {
        nppStatus = nppiDistanceTransformPBAGetBufferSize(oImageSizeROI[i], &nScratchBufferSize);

        if (nppStatus != NPP_NO_ERROR)
        {
            shutDown();
            return -1;
        }

        cudaMalloc((void **) &pScratchDeviceBuffer[(i * 3)], nScratchBufferSize);
        cudaMalloc((void **) &pScratchDeviceBuffer[(i * 3) + 1], nScratchBufferSize);
        cudaMalloc((void **) &pScratchDeviceBuffer[(i * 3) + 2], nScratchBufferSize);
    }

    if (pScratchDeviceBuffer[0] == 0 || pScratchDeviceBuffer[1] == 0 || 
        pScratchDeviceBuffer[2] == 0 || pScratchDeviceBuffer[3] == 0 ||
        pScratchDeviceBuffer[4] == 0 || pScratchDeviceBuffer[5] == 0)
    {
        shutDown();
        return -1;
    }

    cudaMalloc((void **) &pInputImage0_8u_Device, oImageSizeROI[0].width * sizeof(Npp8u) * oImageSizeROI[0].height);

    cudaMalloc((void **) &pInputImage1_8u_Device, oImageSizeROI[1].width * sizeof(Npp8u) * oImageSizeROI[1].height);

    if (pInputImage0_8u_Device == 0 || pInputImage1_8u_Device == 0)
    {
        shutDown();
        return -1;
    }

    cudaMalloc((void **) &pOutputTransformImage0_32f_Device, oImageSizeROI[0].width * sizeof(Npp32f) * oImageSizeROI[0].height);

    cudaMalloc((void **) &pOutputTransformImage1_32f_Device, oImageSizeROI[1].width * sizeof(Npp32f) * oImageSizeROI[1].height);

    cudaMalloc((void **) &pOutputTruncatedImage0_16u_Device, oImageSizeROI[0].width * sizeof(Npp16u) * oImageSizeROI[0].height);

    cudaMalloc((void **) &pOutputTruncatedImage1_16u_Device, oImageSizeROI[1].width * sizeof(Npp16u) * oImageSizeROI[1].height);

    cudaMalloc((void **) &pOutputVoronoiDiagram0_16s_Device, oImageSizeROI[0].width * 2 * sizeof(Npp16s) * oImageSizeROI[0].height);

    cudaMalloc((void **) &pOutputVoronoiDiagram1_16s_Device, oImageSizeROI[1].width * 2 * sizeof(Npp16s) * oImageSizeROI[1].height);

    if (pOutputTransformImage0_32f_Device == 0 || pOutputTransformImage1_32f_Device == 0 ||
        pOutputTruncatedImage0_16u_Device == 0 || pOutputTruncatedImage1_16u_Device == 0 ||
        pOutputVoronoiDiagram0_16s_Device == 0 || pOutputVoronoiDiagram1_16s_Device == 0)
    {
        shutDown();
        return -1;
    }

    // Copy source image0 pixels to device
    if (cudaMemcpy2DAsync(pInputImage0_8u_Device, oImageSizeROI[0].width * sizeof(Npp8u),
                          pInputImage0_8u_Host, oImageSizeROI[0].width * sizeof(Npp8u),
                          oImageSizeROI[0].width * sizeof(Npp8u), oImageSizeROI[0].height,
                          cudaMemcpyHostToDevice, nppStreamCtx.hStream) != cudaSuccess)
    {
        shutDown();
        return -1;
    }

    Npp8u nMinSiteValue = 0;
    Npp8u nMaxSiteValue = 0;


    if (nppiDistanceTransformPBA_8u16u_C1R_Ctx(pInputImage0_8u_Device, oImageSizeROI[0].width * sizeof(Npp8u), nMinSiteValue, nMaxSiteValue, 
                                               0, 0, 
                                               0, 0, 
                                               0, 0,
                                               pOutputTruncatedImage0_16u_Device, oImageSizeROI[0].width * sizeof(Npp16u),
                                               oImageSizeROI[0], pScratchDeviceBuffer[0], nppStreamCtx) != NPP_SUCCESS)
    {
        // Did Cuda fail to launch ?
        cudaError = cudaGetLastError();
        shutDown();
        return -1;
    }
    // Launch was successful.

    // Copy source image1 pixels to device
    if (cudaMemcpy2DAsync(pInputImage1_8u_Device, oImageSizeROI[1].width * sizeof(Npp8u), 
                          pInputImage1_8u_Host, oImageSizeROI[1].width * sizeof(Npp8u), 
                          oImageSizeROI[1].width * sizeof(Npp8u), oImageSizeROI[1].height,
                          cudaMemcpyHostToDevice, nppStreamCtx.hStream) != cudaSuccess)
    {
        shutDown();
        return -1;
    }

    if (nppiDistanceTransformPBA_8u32f_C1R_Ctx(pInputImage1_8u_Device, oImageSizeROI[1].width * sizeof(Npp8u), nMinSiteValue, nMaxSiteValue,
                                               0, 0, 0, 0, 
                                               pOutputVoronoiDiagram1_16s_Device, oImageSizeROI[1].width * 2 * sizeof(Npp16s),
                                               pOutputTransformImage1_32f_Device, oImageSizeROI[1].width * sizeof(Npp32f),
                                               oImageSizeROI[1], pScratchDeviceBuffer[3], nppStreamCtx) != NPP_SUCCESS)
    {
        // Did Cuda fail to launch ?
        cudaError = cudaGetLastError();
        shutDown();
        return -1;
    }

    // Launch was successful.
    if (nppiDistanceTransformPBA_8u16u_C1R_Ctx(pInputImage0_8u_Device, oImageSizeROI[0].width * sizeof(Npp8u), nMinSiteValue, nMaxSiteValue,
                                               pOutputVoronoiDiagram0_16s_Device, oImageSizeROI[0].width * 2 * sizeof(Npp16s),
                                               0, 0,
                                               0, 0,
                                               0, 0,
                                               oImageSizeROI[0], pScratchDeviceBuffer[1], nppStreamCtx) != NPP_SUCCESS)
    {
        // Did Cuda fail to launch ?
        cudaError = cudaGetLastError();
        shutDown();
        return -1;
    }
    // Launch was successful.
    if (nppiDistanceTransformPBA_8u16u_C1R_Ctx(pInputImage1_8u_Device, oImageSizeROI[1].width * sizeof(Npp8u), nMinSiteValue, nMaxSiteValue, 
                                               0, 0,
                                               0, 0,
                                               0, 0,
                                               pOutputTruncatedImage1_16u_Device, oImageSizeROI[1].width * sizeof(Npp16u),
                                               oImageSizeROI[1], pScratchDeviceBuffer[4], nppStreamCtx) != NPP_SUCCESS)
    {
        // Did Cuda fail to launch ?
        cudaError = cudaGetLastError();
        shutDown();
        return -1;
    }
    // Launch was successful.
    if (nppiDistanceTransformPBA_8u32f_C1R_Ctx(pInputImage0_8u_Device, oImageSizeROI[0].width * sizeof(Npp8u), nMinSiteValue, nMaxSiteValue,
                                               0, 0, 0, 0, 0, 0, 
                                               pOutputTransformImage0_32f_Device, oImageSizeROI[0].width * sizeof(Npp32f),
                                               oImageSizeROI[0], pScratchDeviceBuffer[2], nppStreamCtx) != NPP_SUCCESS)
    {
        // Did Cuda fail to launch ?
        cudaError = cudaGetLastError();
        shutDown();
        return -1;
    }
    // Launch was successful.

    // Normally you would continue on with whatever other operations you can do while the images are in device memory.
    // But in this case we are going to copy all of the results back to host memory to save in output files.
    // These copy operations could be interleaved between the NPP calls but you would want to verify that doing so
    // does not decrease overall performance.  
    // These copy operations could also be sent down another Cuda stream but then there is no automatic synchronization
    // so calls to cudaStreamSynchronize and potentially Cuda events would be needed to guarantee proper synchronization.
 
    // Copy back image0 true truncated transform result
    cudaError = cudaMemcpy2DAsync(pOutputTruncatedImage0_16u_Host, oImageSizeROI[0].width * sizeof(Npp16u),
                                  pOutputTruncatedImage0_16u_Device, oImageSizeROI[0].width * sizeof(Npp16u), 
                                  oImageSizeROI[0].width * sizeof(Npp16u), oImageSizeROI[0].height, 
                                  cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

    // Copy back image1 voronoi diagram result
    cudaError = cudaMemcpy2DAsync(pOutputVoronoiDiagram1_16s_Host, oImageSizeROI[1].width * 2 * sizeof(Npp16s),
                                  pOutputVoronoiDiagram1_16s_Device, oImageSizeROI[1].width * 2 * sizeof(Npp16s), 
                                  oImageSizeROI[1].width * 2 * sizeof(Npp16s), oImageSizeROI[1].height, 
                                  cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

    // Copy back image1 true transform result
    cudaError = cudaMemcpy2DAsync(pOutputTransformImage1_32f_Host, oImageSizeROI[1].width * sizeof(Npp32f),
                                  pOutputTransformImage1_32f_Device, oImageSizeROI[1].width * sizeof(Npp32f), 
                                  oImageSizeROI[1].width * sizeof(Npp32f), oImageSizeROI[1].height, 
                                  cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

    // Copy back image0 voronoi diagram result
    cudaError = cudaMemcpy2DAsync(pOutputVoronoiDiagram0_16s_Host, oImageSizeROI[0].width * 2 * sizeof(Npp16s),
                                  pOutputVoronoiDiagram0_16s_Device, oImageSizeROI[0].width * 2 * sizeof(Npp16s), 
                                  oImageSizeROI[0].width * 2 * sizeof(Npp16s), oImageSizeROI[0].height, 
                                  cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

    // Copy back image1 true transform result
    cudaError = cudaMemcpy2DAsync(pOutputTransformImage0_32f_Host, oImageSizeROI[0].width * sizeof(Npp32f),
                                  pOutputTransformImage0_32f_Device, oImageSizeROI[1].width * sizeof(Npp32f), 
                                  oImageSizeROI[1].width * sizeof(Npp32f), oImageSizeROI[1].height, 
                                  cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

    // Copy back image1 true truncated transform result
    cudaError = cudaMemcpy2DAsync(pOutputTruncatedImage1_16u_Host, oImageSizeROI[1].width * sizeof(Npp16u),
                                  pOutputTruncatedImage1_16u_Device, oImageSizeROI[1].width * sizeof(Npp16u), 
                                  oImageSizeROI[1].width * sizeof(Npp16u), oImageSizeROI[1].height, 
                                  cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

    // Since we are saving the result images to files we need to add a synchronization here to make
    // sure that the copies have finished before writing the results to a file.
    if (cudaStreamSynchronize(nppStreamCtx.hStream) != cudaSuccess)
    {
        shutDown();
        return -1;
    }

    FILE * rawOutputFile;
    size_t nSize = 0;

    rawOutputFile = fopen(OutputTruncatedTransformFile0.c_str(), "wb");

    if (rawOutputFile == NULL)
    {
        shutDown();
        return -1;
    }

    nSize = 0;
    for (int j = 0; j < oImageSizeROI[0].height; j++)
    {
        nSize += fwrite(&pOutputTruncatedImage0_16u_Host[j * oImageSizeROI[0].width], sizeof(Npp16u), oImageSizeROI[0].width, rawOutputFile);
    }
    fclose(rawOutputFile);


    rawOutputFile = fopen(OutputVoronoiFile1.c_str(), "wb");

    if (rawOutputFile == NULL)
    {
        shutDown();
        return -1;
    }

    nSize = 0;
    for (int j = 0; j < oImageSizeROI[1].height; j++)
    {
        nSize += fwrite(&pOutputVoronoiDiagram1_16s_Host[j * oImageSizeROI[1].width * 2], 2 * sizeof(Npp16s), 2 * oImageSizeROI[1].width, rawOutputFile);
    }
    fclose(rawOutputFile);


    rawOutputFile = fopen(OutputTrueTransformFile1.c_str(), "wb");

    if (rawOutputFile == NULL)
    {
        shutDown();
        return -1;
    }

    nSize = 0;
    for (int j = 0; j < oImageSizeROI[1].height; j++)
    {
        nSize += fwrite(&pOutputTransformImage1_32f_Host[j * oImageSizeROI[1].width], sizeof(Npp32f), oImageSizeROI[1].width, rawOutputFile);
    }
    fclose(rawOutputFile);


    rawOutputFile = fopen(OutputVoronoiFile0.c_str(), "wb");

    if (rawOutputFile == NULL)
    {
        shutDown();
        return -1;
    }

    nSize = 0;
    for (int j = 0; j < oImageSizeROI[0].height; j++)
    {
        nSize += fwrite(&pOutputVoronoiDiagram0_16s_Host[j * oImageSizeROI[0].width * 2], sizeof(Npp16s), 2 * oImageSizeROI[0].width, rawOutputFile);
    }
    fclose(rawOutputFile);


    rawOutputFile = fopen(OutputTruncatedTransformFile1.c_str(), "wb");

    if (rawOutputFile == NULL)
    {
        shutDown();
        return -1;
    }

    nSize = 0;
    for (int j = 0; j < oImageSizeROI[1].height; j++)
    {
        nSize += fwrite(&pOutputTruncatedImage1_16u_Host[j * oImageSizeROI[1].width], sizeof(Npp16u), oImageSizeROI[1].width, rawOutputFile);
    }
    fclose(rawOutputFile);


    rawOutputFile = fopen(OutputTrueTransformFile0.c_str(), "wb");

    if (rawOutputFile == NULL)
    {
        shutDown();
        return -1;
    }

    nSize = 0;
    for (int j = 0; j < oImageSizeROI[0].height; j++)
    {
        nSize += fwrite(&pOutputTransformImage0_32f_Host[j * oImageSizeROI[0].width], sizeof(Npp32f), oImageSizeROI[0].width, rawOutputFile);
    }
    fclose(rawOutputFile);

    std::cout << "Done!" << std::endl;
    shutDown();

    return 0;
}

