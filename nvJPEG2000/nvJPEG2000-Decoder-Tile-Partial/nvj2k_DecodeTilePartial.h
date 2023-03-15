/*
 * Copyright (c) 2021 - 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include <string.h> // strcmpi

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <windows.h>
#include <filesystem>
const std::string separator = "\\";
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
const std::string separator = "/";
namespace fs = std::experimental::filesystem::v1;
#endif

#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <nvjpeg2k.h>

#define CHECK_CUDA(call)                                                                                          \
    {                                                                                                             \
        cudaError_t _e = (call);                                                                                  \
        if (_e != cudaSuccess)                                                                                    \
        {                                                                                                         \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                                     \
        }                                                                                                         \
    }

#define CHECK_NVJPEG2K(call)                                                                                \
    {                                                                                                       \
        nvjpeg2kStatus_t _e = (call);                                                                       \
        if (_e != NVJPEG2K_STATUS_SUCCESS)                                                                  \
        {                                                                                                   \
            std::cout << "NVJPEG failure: '#" << _e << "' at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return EXIT_FAILURE;                                                                            \
        }                                                                                                   \
    }

typedef std::chrono::high_resolution_clock perfclock;

constexpr int PIPELINE_STAGES = 10;
constexpr int MAX_PRECISION = 16;
constexpr int NUM_COMPONENTS = 4;

typedef struct nvjpeg2kImageSample
{
    nvjpeg2kImageSample():
        pixel_type(NVJPEG2K_UINT8),
        num_comps(0)
    {
        for( int c = 0; c < NUM_COMPONENTS; c++)
        {
            pixel_data[c] = nullptr;
            pitch_in_bytes[c] = 0;
            comp_sz[c] = 0;
        }
    }
    
    void *pixel_data[NUM_COMPONENTS];
    size_t pitch_in_bytes[NUM_COMPONENTS];
    size_t comp_sz[NUM_COMPONENTS];
    nvjpeg2kImageType_t pixel_type;
    uint32_t num_comps;
} nvjpeg2kImageSample_t;

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void *p) { return (int)cudaFree(p); }

int host_malloc(void **p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }

int host_free(void *p) { return (int)cudaFreeHost(p); }

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

struct decode_params_t
{
    std::string input_dir;
    int batch_size;
    int total_images;
    int dev;
    int warmup;

    nvjpeg2kDecodeState_t nvjpeg2k_decode_states[PIPELINE_STAGES];
    nvjpeg2kHandle_t nvjpeg2k_handle;
    cudaStream_t stream[PIPELINE_STAGES];
    std::vector<nvjpeg2kStream_t> jpeg2k_streams;
    bool verbose;
    bool write_decoded;
    std::string output_dir;

    unsigned int win_x0;
    unsigned int win_x1;
    unsigned int win_y0;
    unsigned int win_y1;
    bool partial_decode;
};



int read_next_batch(FileNames &image_names, int batch_size,
                    FileNames::iterator &cur_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &current_names, bool verbose)
{
    int counter = 0;

    while (counter < batch_size)
    {
        if (cur_iter == image_names.end())
        {
            if(verbose)
            {
                std::cerr << "Image list is too short to fill the batch, adding files "
                         "from the beginning of the image list"
                         << std::endl;
            }
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0)
        {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return EXIT_FAILURE;
        }

        // Read an image from disk.
        std::ifstream input(cur_iter->c_str(),
                            std::ios::in | std::ios::binary | std::ios::ate);
        if (!(input.is_open()))
        {
            std::cerr << "Cannot open image: " << *cur_iter
                      << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data[counter].size() < static_cast<size_t>(file_size))
        {
            raw_data[counter].resize(file_size);
        }
        if (!input.read(raw_data[counter].data(), file_size))
        {
            std::cerr << "Cannot read from file: " << *cur_iter
                      << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }
        raw_len[counter] = file_size;
        current_names[counter] = *cur_iter;

        counter++;
        cur_iter++;
    }
    return EXIT_SUCCESS;
}

// *****************************************************************************
// reading input directory to file list
// -----------------------------------------------------------------------------
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    
    if( fs::is_regular_file(sInputPath))
    {
        filelist.push_back(sInputPath);
    }
    else if (fs::is_directory(sInputPath))
    { 
        fs::recursive_directory_iterator iter(sInputPath);
        for(auto& p: iter)
        {
           if( fs::is_regular_file(p))
           {
                filelist.push_back(p.path().string());
           }
        }
    }
    else
    {
        std::cout<<"unable to open input"<<std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// *****************************************************************************
// check for inputDirExists
// -----------------------------------------------------------------------------
int inputDirExists(const char *pathname)
{
    struct stat info;
    if (stat(pathname, &info) != 0)
    {
        return 0; // Directory does not exists
    }
    else if (info.st_mode & S_IFDIR)
    {
        // is a directory
        return 1;
    }
    else
    {
        // is not a directory
        return 0;
    }
}

// *****************************************************************************
// check for getInputDir
// -----------------------------------------------------------------------------
int getInputDir(std::string &input_dir, const char *executable_path)
{
    int found = 0;
    if (executable_path != 0)
    {
        std::string executable_name = std::string(executable_path);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);

        if (executable_name.rfind(".exe") != std::string::npos)
        {
            // we strip .exe, only if the .exe is found
            executable_name.resize(executable_name.size() - 4);
        }
#else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0, delimiter_pos + 1);
#endif

        // Search in default paths for input images.
        std::string pathname = "";
        const char *searchPath[] = {
            "./images"};

        for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i)
        {
            std::string pathname(searchPath[i]);
            size_t executable_name_pos = pathname.find("<executable_name>");

            // If there is executable_name variable in the searchPath
            // replace it with the value
            if (executable_name_pos != std::string::npos)
            {
                pathname.replace(executable_name_pos, strlen("<executable_name>"),
                                 executable_name);
            }

            if (inputDirExists(pathname.c_str()))
            {
                input_dir = pathname + "/";
                found = 1;
                break;
            }
        }
    }
    return found;
}

// write PGM, input - single channel, device
template <typename D>
int writePGM(const char *filename, const D *pSrc, size_t nSrcStep, int nWidth, int nHeight, uint8_t precision, uint8_t sgn)
{
    std::ofstream rOutputStream(filename, std::fstream::binary);
    if (!rOutputStream)
    {
        std::cerr << "Cannot open output file: " << filename << std::endl;
        return EXIT_FAILURE;
    }
    std::vector<D> img(nHeight * (nSrcStep / sizeof(D)));
    D *hpSrc = img.data();

    CHECK_CUDA(cudaMemcpy2D(hpSrc, nSrcStep, pSrc, nSrcStep, nWidth * sizeof(D), nHeight, cudaMemcpyDeviceToHost));

    rOutputStream << "P5\n";
    rOutputStream << "#nvJPEG2000\n";
    rOutputStream << nWidth << " " << nHeight << "\n";
    rOutputStream << (1 << precision) - 1 << "\n";

    D *pTemp = hpSrc;
    const D *pEndRow = pTemp + nHeight * (nSrcStep / sizeof(D));
    for (; pTemp < pEndRow; pTemp += (nSrcStep / sizeof(D)))
    {
        const D *pRow = pTemp;
        const D *pEndColumn = pRow + nWidth;
        for (; pRow < pEndColumn; ++pRow)
        {
            if (precision <= 8)
            {
                rOutputStream << static_cast<unsigned char>(*pRow);
            }
            else if (precision <= 16)
            {
                int pix_val = *pRow;
                if(sgn)
                {
                    pix_val += (1 << (precision - 1));
                    if (pix_val > 65535) 
                    {
                        pix_val = 65535;
                    } 
                    else if (pix_val < 0) 
                    {
                        pix_val = 0;
                    }
                }
                rOutputStream << static_cast<unsigned char>((pix_val) >> 8) << static_cast<unsigned char>((pix_val) & 0xff);
            }
        }
    }
    return 0;
}

// write bmp, input - RGB, device
template <typename D>
int writeBMP(const char *filename,
             const D *d_chanR, size_t pitchR,
             const D *d_chanG, size_t pitchG,
             const D *d_chanB, size_t pitchB,
             int width, int height,
             uint8_t precision,
             bool verbose)
{

    unsigned int headers[13];
    FILE *outfile;
    int extrabytes;
    int paddedsize;
    int x;
    int y;
    int n;
    int red, green, blue;

    std::vector<D> vchanR(height * width);
    std::vector<D> vchanG(height * width);
    std::vector<D> vchanB(height * width);
    D *chanR = vchanR.data();
    D *chanG = vchanG.data();
    D *chanB = vchanB.data();
    CHECK_CUDA(cudaMemcpy2D(chanR, (size_t)width * sizeof(D), d_chanR, pitchR, width * sizeof(D), height, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaMemcpy2D(chanG, (size_t)width * sizeof(D), d_chanG, pitchG, width * sizeof(D), height, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaMemcpy2D(chanB, (size_t)width * sizeof(D), d_chanB, pitchB, width * sizeof(D), height, cudaMemcpyDeviceToHost));

    extrabytes = 4 - ((width * 3) % 4); // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extrabytes == 4)
        extrabytes = 0;

    paddedsize = ((width * 3) + extrabytes) * height;

    headers[0] = paddedsize + 54; // bfSize (whole file size)
    headers[1] = 0;               // bfReserved (both)
    headers[2] = 54;              // bfOffbits
    headers[3] = 40;              // biSize
    headers[4] = width;           // biWidth
    headers[5] = height;          // biHeight

  

    headers[7] = 0;          // biCompression
    headers[8] = paddedsize; // biSizeImage
    headers[9] = 0;          // biXPelsPerMeter
    headers[10] = 0;         // biYPelsPerMeter
    headers[11] = 0;         // biClrUsed
    headers[12] = 0;         // biClrImportant

    if (!(outfile = fopen(filename, "wb")))
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }

    fprintf(outfile, "BM");

    for (n = 0; n <= 5; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
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
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }
    
    if (verbose && precision > 8)
    {
        std::cout<<"BMP write - truncating "<< (int)precision <<" bit data to 8 bit"<<std::endl;
    }

    //
    // Headers done, now write the data...
    //
    for (y = height - 1; y >= 0; y--) // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++)
        {
            red = chanR[y * width + x];
            green = chanG[y * width + x];
            blue = chanB[y * width + x];

            int scale = precision - 8;
            if (scale > 0) 
            {
                red = ((red >> scale) + ((red >> (scale - 1)) % 2));
                green = ((green >> scale) + ((green >> (scale - 1)) % 2));
                blue = ((blue >> scale) + ((blue >> (scale - 1)) % 2));
            }


            if (red > 255)
                red = 255;
            if (red < 0)
                red = 0;
            if (green > 255)
                green = 255;
            if (green < 0)
                green = 0;
            if (blue > 255)
                blue = 255;
            if (blue < 0)
                blue = 0;
            // Also, it's written in (b,g,r) format...

            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++)
            {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1)
    {
        return index;
    }
    else
    {
        std::cout << "Error, parameter " << parm
                  << " has been specified more than once, exiting\n"
                  << std::endl;
        return -1;
    }

    return -1;
}
