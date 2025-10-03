/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Note:  If you want to view these images we HIGHLY recommend using imagej which is free on the internet and works on most platforms. 
//
//        The files read and written by this sample app use RAW image format, that is, only the image data itself exists in the files
//        with no image format information.   When viewing RAW files with imagej just enter the image size and bit depth/type values that
//        are part of the file name when requested by imagej.
// 

#define NUMBER_OF_IMAGES 2

    Npp8u  * pInputImageDev[NUMBER_OF_IMAGES];
    Npp8u  * pInputImageHost[NUMBER_OF_IMAGES];
    Npp8u  * pScratchBufferDev[NUMBER_OF_IMAGES];
    Npp64f * pDistanceTransformImageDev[NUMBER_OF_IMAGES];
    Npp64f * pDistanceTransformImageHost[NUMBER_OF_IMAGES];
    Npp16u * pDistanceTransformImageHost_16u[NUMBER_OF_IMAGES];
    Npp64f * pSignedInputImageDev[NUMBER_OF_IMAGES];
    Npp8u  * pSignedInputImageHost[NUMBER_OF_IMAGES];
    Npp64f * pSignedDistanceTransformImageDev[NUMBER_OF_IMAGES];
    Npp64f * pSignedDistanceTransformImageHost[NUMBER_OF_IMAGES];
    Npp16u * pSignedDistanceTransformImageHost_16u[NUMBER_OF_IMAGES];
    Npp8u  * pSignedScratchBufferDev;

void tearDown() // Clean up and tear down
{

    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        if (pScratchBufferDev[j] != 0)
            cudaFree(pScratchBufferDev[j]);
        if (pDistanceTransformImageDev[j] != 0)
            cudaFree(pDistanceTransformImageDev[j]);
        if (pInputImageDev[j] != 0)
            cudaFree(pInputImageDev[j]);
        if (pDistanceTransformImageHost_16u[j] != 0)
        {
            free(pDistanceTransformImageHost_16u[j]);
            pDistanceTransformImageHost_16u[j] = 0;
        }
        if (pDistanceTransformImageHost[j] != 0)
        {
            free(pDistanceTransformImageHost[j]);
            pDistanceTransformImageHost[j] = 0;
        }
        if (pInputImageHost[j] != 0)
            free(pInputImageHost[j]);
        if (pSignedDistanceTransformImageDev[j] != 0)
            cudaFree(pSignedDistanceTransformImageDev[j]);
        if (pSignedInputImageDev[j] != 0)
        {
            cudaFree(pSignedInputImageDev[j]);
            pSignedInputImageDev[j] = 0;
        }
        if (pSignedDistanceTransformImageHost_16u[j] != 0)
        {
            free(pSignedDistanceTransformImageHost_16u[j]);
            pSignedDistanceTransformImageHost_16u[j] = 0;
        }
        if (pSignedDistanceTransformImageHost[j] != 0)
        {
            free(pSignedDistanceTransformImageHost[j]);
            pSignedDistanceTransformImageHost[j] = 0;
        }
        if (pSignedInputImageHost[j] != 0)
            free(pSignedInputImageHost[j]);
    }
    if (pSignedScratchBufferDev != 0)
        cudaFree(pSignedScratchBufferDev);
}

const std::string & Path = std::string("../images");

const std::string & InputFile0 = Path + std::string("DistanceSampler_512x512_8u.raw");
const std::string & InputFile1 = Path + std::string("DistanceSampler_512x512_Inverted_8u.raw");
const std::string & SignedInputFile0 = Path + std::string("SignedCircle_256x206_64f.raw");
const std::string & SignedInputFile1 = Path + std::string("SignedCircle_256x206_Inverted_64f.raw");
const std::string & SignedInputFile2 = Path + std::string("SignedLith_554x554_32f.raw");

const std::string & DistanceSamplerTransformFile0 = Path + std::string("DistanceSamplerTransform_512x512_64f.raw");
const std::string & DistanceSamplerTransform_Inverted_File1 = Path + std::string("DistanceSamplerTransform_512x512_Inverted_64f.raw");
const std::string & DistanceSamplerTransform_16u_File0 = Path + std::string("DistanceSamplerTransform_512x512_16u.raw");
const std::string & DistanceSamplerTransform_16u_Inverted_File1 = Path + std::string("DistanceSamplerTransform_512x512_Inverted_16u.raw");
const std::string & SignedDistanceCircleTransform_64f_File0 = Path + std::string("SignedDistanceCircleTransform_256x206_64f.raw");
const std::string & SignedDistanceCircleTransform_16u_File0 = Path + std::string("SignedDistanceCircleTransform_256x206_16u.raw");
const std::string & SignedDistanceCircleTransform_64f_File1 = Path + std::string("SignedDistanceCircleTransform_256x206_Inverted_64f.raw");
const std::string & SignedDistanceCircleTransform_16u_File1 = Path + std::string("SignedDistanceCircleTransform_256x206_Inverted_16u.raw");
const std::string & SignedDistanceLithTransform_64f_File0 = Path + std::string("SignedDistanceLithTransform_554x554_64f.raw");
const std::string & SignedDistanceLithTransform_16u_File0 = Path + std::string("SignedDistanceLithTransform_554x554_16u.raw");
const std::string & SignedDistanceLithTransformIndices_16s_File0 = Path + std::string("SignedDistanceLithTransformVoronoiIndices_554x554_16s.raw");
const std::string & SignedDistanceLithTransformManhattan_16s_File0 = Path + std::string("SignedDistanceLithTransformVoronoiRelativeManhattan_554x554_16s.raw");

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
loadRaw32fImage(Npp8u * pImage, int nWidth, int nHeight, int nImage)
{
    FILE * bmpFile;
    size_t nSize;

    if (nImage == 0)
	{
        if (nWidth != 554 || nHeight != 554) 
            return -1;
        fopen_s(&bmpFile, SignedInputFile2.c_str(), "rb");
    }
    else
    {
        printf ("Input file load failed.\n");
        return -1;
    }

    if (bmpFile == NULL) 
        return -1;
    nSize = fread(pImage, 1, nWidth * sizeof(Npp32f) * nHeight, bmpFile);
    if (nSize < nWidth * sizeof(Npp32f) * nHeight)
    {
        fclose(bmpFile);        
        return -1;
    }
    fclose(bmpFile);

    printf ("Input file load succeeded.\n");

    return 0;
}

int 
loadRaw64fImage(Npp8u * pImage, int nWidth, int nHeight, int nImage)
{
    FILE * bmpFile;
    size_t nSize;

    if (nImage == 0)
	{
        if (nWidth != 256 || nHeight != 206) 
            return -1;
        fopen_s(&bmpFile, SignedInputFile0.c_str(), "rb");
    }
    else if (nImage == 1)
    {
        if (nWidth != 256 || nHeight != 206) 
            return -1;
        fopen_s(&bmpFile, SignedInputFile1.c_str(), "rb");
    }
    else
    {
        printf ("Signed input file load failed.\n");
        return -1;
    }

    if (bmpFile == NULL) 
        return -1;
    nSize = fread(pImage, 1, nWidth * sizeof(Npp64f) * nHeight, bmpFile);
    if (nSize < nWidth * sizeof(Npp64f) * nHeight)
    {
        fclose(bmpFile);        
        return -1;
    }
    fclose(bmpFile);

    printf ("Signed input file load succeeded.\n");

    return 0;
}



int 
main( int argc, char** argv )
{
    cudaError_t cudaError;
    NppStatus nppStatus;
    NppStreamContext nppStreamCtx;
	FILE * bmpFile;

    for (int j = 0; j < NUMBER_OF_IMAGES; j++)
    {
        pInputImageDev[j] = 0;
        pInputImageHost[j] = 0;
        pScratchBufferDev[j] = 0;
        pDistanceTransformImageDev[j] = 0;
        pDistanceTransformImageHost[j] = 0;
        pDistanceTransformImageHost_16u[j] = 0;
        pSignedInputImageDev[j] = 0;
        pSignedInputImageHost[j] = 0;
        pSignedDistanceTransformImageDev[j] = 0;
        pSignedDistanceTransformImageHost[j] = 0;
        pSignedDistanceTransformImageHost_16u[j] = 0;
    }
    pSignedScratchBufferDev = 0;

    // You MUST create at least one NPP stream context.

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

    // Process unsigned distance transform images first.

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
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

        // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can significantly improve the performance of many NPP functions, 
        // for UF function label markers generation or compression DO NOT USE cudaMallocPitch().  Doing so could result in incorrect output.

        cudaError = cudaMalloc ((void**)&pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void**)&pDistanceTransformImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pInputImageHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp8u) * oSizeROI[nImage].height));
        pDistanceTransformImageHost[nImage] = reinterpret_cast<Npp64f *>(malloc(oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height));
        pDistanceTransformImageHost_16u[nImage] = reinterpret_cast<Npp16u *>(malloc(oSizeROI[nImage].width * sizeof(Npp16u) * oSizeROI[nImage].height));

        size_t nScratchBufferSize;

        // Get base distance transform scratch device memory buffer size
        nppStatus = nppPlusV::nppiDistanceTransformPBAGetBufferSize(oSizeROI[nImage], &nScratchBufferSize);

        cudaError = cudaMalloc ((void **)&pScratchBufferDev[nImage], nScratchBufferSize);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        if (loadRaw8BitImage(pInputImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, nImage) == 0)
        {
            cudaError = cudaMemcpy2DAsync(pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), pInputImageHost[nImage], 
                                                          oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].width * sizeof(Npp8u), oSizeROI[nImage].height, 
                                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // Using nppPlusV:: namespace with NPPPlus calls will lock your executable into the specific NPPPlus release it was built with.
            nppStatus = nppPlusV::nppiDistanceTransformPBA_8u64f_C1R_Ctx(pInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp8u), 0, 0, 
                                                                         0, 0, 0, 0, 0, 0,
                                                                         pDistanceTransformImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f),
                                                                         oSizeROI[nImage], pScratchBufferDev[nImage], 0, nppStreamCtx);
            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("DistanceSamplerTransform_512x512_64f failed.\n");
                else if (nImage == 1)
                    printf("DistanceSamplerTransform_512x512_Inverted_64f failed.\n");
                tearDown();
                return -1;
            }

            cudaError = cudaMemcpy2DAsync(pDistanceTransformImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp64f), 
                                          pDistanceTransformImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f), 
                                          oSizeROI[nImage].width * sizeof(Npp64f), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait for host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post processing cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }

            if (nImage == 0)
                fopen_s(&bmpFile, DistanceSamplerTransformFile0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, DistanceSamplerTransform_Inverted_File1.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            size_t nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pDistanceTransformImageHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp64f), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            // Convert output files to something more easily viewable

            for (int i = 0; i < oSizeROI[nImage].width * oSizeROI[nImage].height; i++)
            {
                pDistanceTransformImageHost_16u[nImage][i] = static_cast<Npp16u>(pDistanceTransformImageHost[nImage][i]);
            }

            if (nImage == 0)
                fopen_s(&bmpFile, DistanceSamplerTransform_16u_File0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, DistanceSamplerTransform_16u_Inverted_File1.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pDistanceTransformImageHost_16u[nImage][j * oSizeROI[nImage].width], sizeof(Npp16u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);
        }
    }


    // Now process signed distance transform images.

    for (int nImage = 0; nImage < NUMBER_OF_IMAGES; nImage++)
    {
        oSizeROI[nImage].width = 256;
        oSizeROI[nImage].height = 206;

        // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can significantly improve the performance of many NPP functions, 
        // for UF function label markers generation or compression DO NOT USE cudaMallocPitch().  Doing so could result in incorrect output.

        cudaError = cudaMalloc ((void**)&pSignedInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        cudaError = cudaMalloc ((void**)&pSignedDistanceTransformImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        pSignedInputImageHost[nImage] = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height));
        pSignedDistanceTransformImageHost[nImage] = reinterpret_cast<Npp64f *>(malloc(oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height));
        pSignedDistanceTransformImageHost_16u[nImage] = reinterpret_cast<Npp16u *>(malloc(oSizeROI[nImage].width * sizeof(Npp16u) * oSizeROI[nImage].height));

        size_t nSignedScratchBufferSize;

        // Get base distance transform scratch device memory buffer size

        // Using nppPlusV:: namespace with NPPPlus calls will lock your executable into the specific NPPPlus release it was built with.
        nppStatus = nppPlusV::nppiSignedDistanceTransformPBAGet64fBufferSize(oSizeROI[nImage], &nSignedScratchBufferSize);

        cudaError = cudaMalloc ((void **)&pSignedScratchBufferDev, nSignedScratchBufferSize);
        if (cudaError != cudaSuccess)
            return NPP_MEMORY_ALLOCATION_ERR;

        if (loadRaw64fImage(pSignedInputImageHost[nImage], oSizeROI[nImage].width, oSizeROI[nImage].height, nImage) == 0)
        {
            // Convert from 0.0 to 1.0 to -1.0 to 1.0
            Npp64f * pTmpHost = reinterpret_cast<Npp64f *>(pSignedInputImageHost[nImage]);

            for (int i = 0; i < oSizeROI[nImage].width * oSizeROI[nImage].height; ++i)
            {
                pTmpHost[i] = (pTmpHost[i] - 0.5) * 2.0;
            }

            cudaError = cudaMemcpy2DAsync(pSignedInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f), pSignedInputImageHost[nImage],
                                                          oSizeROI[nImage].width * sizeof(Npp64f), oSizeROI[nImage].width * sizeof(Npp64f), oSizeROI[nImage].height, 
                                                          cudaMemcpyHostToDevice, nppStreamCtx.hStream);

            // Using nppPlusV:: namespace with NPPPlus calls will lock your executable into the specific NPPPlus release it was built with.
            nppStatus = nppPlusV::nppiSignedDistanceTransformPBA_64f_C1R_Ctx(pSignedInputImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f), 0.0, 0.0, 0.0, 
                                                                             0, 0, 0, 0, 0, 0,
                                                                             pSignedDistanceTransformImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f),
                                                                             oSizeROI[nImage], pSignedScratchBufferDev, 0, nppStreamCtx);
            if (nppStatus != NPP_SUCCESS)  
            {
                if (nImage == 0)
                    printf("SignedDistanceCircleTransform_256x206_64f failed.\n");
                else if (nImage == 1)
                    printf("SignedDistanceCircleTransform_256x206_Inverted_64f failed.\n");
                tearDown();
                return -1;
            }

            cudaError = cudaMemcpy2DAsync(pSignedDistanceTransformImageHost[nImage], oSizeROI[nImage].width * sizeof(Npp64f), 
                                          pSignedDistanceTransformImageDev[nImage], oSizeROI[nImage].width * sizeof(Npp64f), 
                                          oSizeROI[nImage].width * sizeof(Npp64f), oSizeROI[nImage].height,
                                          cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

            // Wait for host image read backs to complete, not necessary if no need to synchronize
            if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
            {
                printf ("Post processing cudaStreamSynchronize failed\n");
                tearDown();
                return -1;
            }


            if (nImage == 0)
                fopen_s(&bmpFile, SignedDistanceCircleTransform_64f_File0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, SignedDistanceCircleTransform_64f_File1.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            size_t nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSignedDistanceTransformImageHost[nImage][j * oSizeROI[nImage].width], sizeof(Npp64f), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);

            // Convert output files to something more easily viewable

            for (int i = 0; i < oSizeROI[nImage].width * oSizeROI[nImage].height; i++)
            {
                pSignedDistanceTransformImageHost_16u[nImage][i] = static_cast<Npp16u>(pSignedDistanceTransformImageHost[nImage][i] + 32700.0);
            }

            if (nImage == 0)
                fopen_s(&bmpFile, SignedDistanceCircleTransform_16u_File0.c_str(), "wb");
            else if (nImage == 1)
                fopen_s(&bmpFile, SignedDistanceCircleTransform_16u_File1.c_str(), "wb");

            if (bmpFile == NULL) 
                return -1;
            nSize = 0;
            for (int j = 0; j < oSizeROI[nImage].height; j++)
            {
                nSize += fwrite(&pSignedDistanceTransformImageHost_16u[nImage][j * oSizeROI[nImage].width], sizeof(Npp16u), oSizeROI[nImage].width, bmpFile);
            }
            fclose(bmpFile);
        }
    }

    tearDown();

    // Process 32f signed distance transform image

    int nImage = 0;

    Npp32f * pSignedLithInputImageDev = 0;
    Npp8u  * pSignedLithInputImageHost = 0;
    Npp64f * pSignedLithDistanceTransformImageDev = 0;
    Npp64f * pSignedLithDistanceTransformImageHost = 0;
    Npp16u * pSignedLithDistanceTransformImageHost_16u = 0;
    Npp16s * pOutputIndicesDev = 0;
    Npp16s * pOutputManhattanDev = 0;
    Npp16s * pOutputIndicesHost = 0;
    Npp16s * pOutputManhattanHost = 0;

    oSizeROI[nImage].width = 554;
    oSizeROI[nImage].height = 554;

    // NOTE: While using cudaMallocPitch() to allocate device memory for NPP can significantly improve the performance of many NPP functions, 
    // for UF function label markers generation or compression DO NOT USE cudaMallocPitch().  Doing so could result in incorrect output.

    cudaError = cudaMalloc ((void**)&pSignedLithInputImageDev, oSizeROI[nImage].width * sizeof(Npp32f) * oSizeROI[nImage].height);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    cudaError = cudaMalloc ((void**)&pSignedLithDistanceTransformImageDev, oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    cudaError = cudaMalloc((void **) &pOutputIndicesDev, oSizeROI[nImage].width * 2 * sizeof(Npp16s) * oSizeROI[nImage].height);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    cudaError = cudaMalloc((void **) &pOutputManhattanDev, oSizeROI[nImage].width * 2 * sizeof(Npp16s) * oSizeROI[nImage].height);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    pSignedLithInputImageHost = reinterpret_cast<Npp8u *>(malloc(oSizeROI[nImage].width * sizeof(Npp32f) * oSizeROI[nImage].height));
    pSignedLithDistanceTransformImageHost = reinterpret_cast<Npp64f *>(malloc(oSizeROI[nImage].width * sizeof(Npp64f) * oSizeROI[nImage].height));
    pSignedLithDistanceTransformImageHost_16u = reinterpret_cast<Npp16u *>(malloc(oSizeROI[nImage].width * sizeof(Npp16u) * oSizeROI[nImage].height));
    pOutputIndicesHost = reinterpret_cast<Npp16s *>(malloc(oSizeROI[nImage].width * 2 * sizeof(Npp16s) * oSizeROI[nImage].height));
    pOutputManhattanHost = reinterpret_cast<Npp16s *>(malloc(oSizeROI[nImage].width * 2 * sizeof(Npp16s) * oSizeROI[nImage].height));

    size_t nSignedScratchBufferSize;

    // Get base distance transform scratch device memory buffer size

    // Using nppPlusV:: namespace with NPPPlus calls will lock your executable into the specific NPPPlus release it was built with.
    nppStatus = nppPlusV::nppiSignedDistanceTransformPBAGet64fBufferSize(oSizeROI[nImage], &nSignedScratchBufferSize);

    cudaError = cudaMalloc ((void **)&pSignedScratchBufferDev, nSignedScratchBufferSize);
    if (cudaError != cudaSuccess)
        return NPP_MEMORY_ALLOCATION_ERR;

    if (loadRaw32fImage(pSignedLithInputImageHost, oSizeROI[nImage].width, oSizeROI[nImage].height, nImage) == 0)
    {

        // This input image is already signed.

        cudaError = cudaMemcpy2DAsync(pSignedLithInputImageDev, oSizeROI[nImage].width * sizeof(Npp32f), pSignedLithInputImageHost,
                                                      oSizeROI[nImage].width * sizeof(Npp32f), oSizeROI[nImage].width * sizeof(Npp32f), oSizeROI[nImage].height, 
                                                      cudaMemcpyHostToDevice, nppStreamCtx.hStream);

        // Using nppPlusV:: namespace with NPPPlus calls will lock your executable into the specific NPPPlus release it was built with.
        nppStatus = nppPlusV::nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx(pSignedLithInputImageDev, oSizeROI[nImage].width * sizeof(Npp32f), 0.1f, 0.0, 0.0, 
                                                                            0, 0,
                                                                            pOutputIndicesDev, oSizeROI[nImage].width * 2 * sizeof(Npp16s),
                                                                            pOutputManhattanDev, oSizeROI[nImage].width * 2 * sizeof(Npp16s),
                                                                            pSignedLithDistanceTransformImageDev, oSizeROI[nImage].width * sizeof(Npp64f),
                                                                            oSizeROI[nImage], pSignedScratchBufferDev, 0, nppStreamCtx);
        if (nppStatus != NPP_SUCCESS)  
        {
            if (nImage == 0)
                printf("SignedDistanceLithTransform_554x554_64f failed.\n");
            tearDown();
            return -1;
        }

        // If copying results back to host wait for transform function to complete.
        cudaError = cudaStreamSynchronize(nppStreamCtx.hStream); 

        // Copy back the trnasform result
        cudaError = cudaMemcpy2DAsync(pSignedLithDistanceTransformImageHost, oSizeROI[nImage].width * sizeof(Npp64f), 
                                      pSignedLithDistanceTransformImageDev, oSizeROI[nImage].width * sizeof(Npp64f), 
                                      oSizeROI[nImage].width * sizeof(Npp64f), oSizeROI[nImage].height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream);

        // Copy back some of the voronoi results
        cudaError = cudaMemcpy2DAsync(pOutputIndicesHost, oSizeROI[nImage].width * 2 * sizeof(Npp16s),
                                      pOutputIndicesDev, oSizeROI[nImage].width * 2 * sizeof(Npp16s), 
                                      oSizeROI[nImage].width * 2 * sizeof(Npp16s), oSizeROI[nImage].height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

        cudaError = cudaMemcpy2DAsync(pOutputManhattanHost, oSizeROI[nImage].width * 2 * sizeof(Npp16s),
                                      pOutputManhattanDev, oSizeROI[nImage].width * 2 * sizeof(Npp16s), 
                                      oSizeROI[nImage].width * 2 * sizeof(Npp16s), oSizeROI[nImage].height,
                                      cudaMemcpyDeviceToHost, nppStreamCtx.hStream); 

        // Wait for host image read backs to complete, not necessary if no particular need to synchronize
        if ((cudaError = cudaStreamSynchronize(nppStreamCtx.hStream)) != cudaSuccess) 
        {
            printf ("Post processing cudaStreamSynchronize failed\n");
            tearDown();
            return -1;
        }


        if (nImage == 0)
            fopen_s(&bmpFile, SignedDistanceLithTransform_64f_File0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        size_t nSize = 0;
        for (int j = 0; j < oSizeROI[nImage].height; j++)
        {
            nSize += fwrite(&pSignedLithDistanceTransformImageHost[j * oSizeROI[nImage].width], sizeof(Npp64f), oSizeROI[nImage].width, bmpFile);
        }
        fclose(bmpFile);

        // Convert output files to something more easily viewable

        if (nImage == 0)
            fopen_s(&bmpFile, SignedDistanceLithTransformIndices_16s_File0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI[nImage].height; j++)
        {
            nSize += fwrite(&pOutputIndicesHost[j * oSizeROI[nImage].width * 2], sizeof(Npp16s), oSizeROI[nImage].width * 2, bmpFile);
        }
        fclose(bmpFile);


        if (nImage == 0)
            fopen_s(&bmpFile, SignedDistanceLithTransformManhattan_16s_File0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI[nImage].height; j++)
        {
            nSize += fwrite(&pOutputManhattanHost[j * oSizeROI[nImage].width * 2], sizeof(Npp16s), oSizeROI[nImage].width * 2, bmpFile);
        }
        fclose(bmpFile);


        for (int i = 0; i < oSizeROI[nImage].width * oSizeROI[nImage].height; i++)
        {
            pSignedLithDistanceTransformImageHost_16u[i] = static_cast<Npp16u>(pSignedLithDistanceTransformImageHost[i] + 32700.0);
        }

        if (nImage == 0)
            fopen_s(&bmpFile, SignedDistanceLithTransform_16u_File0.c_str(), "wb");

        if (bmpFile == NULL) 
            return -1;
        nSize = 0;
        for (int j = 0; j < oSizeROI[nImage].height; j++)
        {
            nSize += fwrite(&pSignedLithDistanceTransformImageHost_16u[j * oSizeROI[nImage].width], sizeof(Npp16u), oSizeROI[nImage].width, bmpFile);
        }
        fclose(bmpFile);

    }

    cudaFree(pSignedLithInputImageDev);
    free(pSignedLithInputImageHost);
    cudaFree(pSignedLithDistanceTransformImageDev);
    free(pSignedLithDistanceTransformImageHost);
    free(pSignedLithDistanceTransformImageHost_16u);
    cudaFree(pOutputIndicesDev);
    cudaFree(pOutputManhattanDev);
    free(pOutputIndicesHost);
    free(pOutputManhattanHost);

    return 0;
}


