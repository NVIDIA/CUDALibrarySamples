/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef __UTILS_H__
#define __UTILS_H__
#include <sys/types.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_MSC_VER)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#include <windows.h>
#include <chrono>
#  pragma warning(disable:4819)
#else
#include <unistd.h>
#include <time.h>
#endif
#ifdef __cplusplus
#define UTILS_LINKAGE "C"
#else
#define UTILS_LINKAGE
#endif

extern UTILS_LINKAGE void *Malloc(size_t sz);
extern UTILS_LINKAGE void Free(void **ptr);
extern UTILS_LINKAGE void *Realloc(void *ptr, size_t sz);
extern UTILS_LINKAGE FILE *Fopen(const char *path, const char *mode);
extern UTILS_LINKAGE size_t Fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
extern UTILS_LINKAGE size_t Fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
extern UTILS_LINKAGE int Remove(const char *pathname);
extern UTILS_LINKAGE off_t getFsize(const char *fpath);
extern UTILS_LINKAGE double Wtime(void);

#endif
