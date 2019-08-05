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

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <string.h>  // strcmpi
#ifndef _WIN64
#include <sys/time.h>  // timings
#include <unistd.h>
#endif
#include <dirent.h>  
#include <sys/stat.h>
#include <sys/types.h>


#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <nppi_geometry_transforms.h>


#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

struct image_resize_params_t {
  std::string input_dir;
  std::string output_dir;
  int quality;
  int width;
  int height;
  int dev;
};


typedef struct {
    NppiSize size;
    nvjpegImage_t data;
} image_t;


int dev_malloc(void** p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

int dev_free(void* p)
{
    return (int)cudaFree(p);
}

bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}


// *****************************************************************************
// reading input directory to file list
// -----------------------------------------------------------------------------
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    int error_code = 1;
    struct stat s;

    if( stat(sInputPath.c_str(), &s) == 0 )
    {
        if( s.st_mode & S_IFREG )
        {
            filelist.push_back(sInputPath);
        }
        else if( s.st_mode & S_IFDIR )
        {
            // processing each file in directory
            DIR *dir_handle;
            struct dirent *dir;
            dir_handle = opendir(sInputPath.c_str());
            std::vector<std::string> filenames;
            if (dir_handle)
            {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL)
                {
                    if (dir->d_type == DT_REG)
                    {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    }
                    else if (dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..")
                        {
                            readInput(sInputPath + sname + "/", filelist);
                        }
                    }
                }
                closedir(dir_handle);
            }
            else
            {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        }
        else
        {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    }
    else
    {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }

    return 0;
}

// *****************************************************************************
// check for inputDirExists
// -----------------------------------------------------------------------------
int inputDirExists(const char *pathname) {
  struct stat info;
  if (stat(pathname, &info) != 0) {
    return 0;  // Directory does not exists
  } else if (info.st_mode & S_IFDIR) {
    // is a directory
    return 1;
  } else {
    // is not a directory
    return 0;
  }
}

// *****************************************************************************
// check for getInputDir
// -----------------------------------------------------------------------------
int getInputDir(std::string &input_dir, const char *executable_path) {
  int found = 0;
  if (executable_path != 0) {
    std::string executable_name = std::string(executable_path);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    // Windows path delimiter
    size_t delimiter_pos = executable_name.find_last_of('\\');
    executable_name.erase(0, delimiter_pos + 1);

    if (executable_name.rfind(".exe") != std::string::npos) {
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

    for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i) {
      std::string pathname(searchPath[i]);
      size_t executable_name_pos = pathname.find("<executable_name>");

      // If there is executable_name variable in the searchPath
      // replace it with the value
      if (executable_name_pos != std::string::npos) {
        pathname.replace(executable_name_pos, strlen("<executable_name>"),
                         executable_name);
      }

      if (inputDirExists(pathname.c_str())) {
        input_dir = pathname + "/";
        found = 1;
        break;
      }
    }
  }
  return found;
}

// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char **argv, int argc, const char *parm) {
  int count = 0;
  int index = -1;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], parm, 100) == 0) {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1) {
    return index;
  } else {
    std::cout << "Error, parameter " << parm
              << " has been specified more than once, exiting\n"
              << std::endl;
    return -1;
  }

  return -1;
}
