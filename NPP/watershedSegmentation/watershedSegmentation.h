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

#include <npp.h>


#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif


struct image_watershedsegmentation_params_t {
  int numofbatch;
  int dev;
};


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
