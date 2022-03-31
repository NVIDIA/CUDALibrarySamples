/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "nvTiff_utils.h"

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

void *Malloc(size_t sz) {

	void *ptr;

	if (!sz) {
		printf("Allocating zero bytes...\n");
		exit(EXIT_FAILURE);
	}
	ptr = (void *)malloc(sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	memset(ptr, 0, sz);
	return ptr;
}

void Free(void **ptr) {

	if (*ptr) {
		free(*ptr);
		*ptr = NULL;
	}
	return;
}

void *Realloc(void *ptr, size_t sz) {

        void *lp;

	if (!sz) {
		printf("Re-allocating to zero bytes, are you sure you want this?\n");
	}
        lp = (void *)realloc(ptr, sz);
        if (!lp && sz) {
                fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
                exit(EXIT_FAILURE);
        }
        return lp;
}

FILE *Fopen(const char *path, const char *mode) {

        FILE *fp = NULL;
        fp = fopen(path, mode);
        if (!fp) {
                fprintf(stderr, "Cannot open file %s...\n", path);
                exit(EXIT_FAILURE);
        }
        return fp;
}

size_t Fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {

	size_t wmemb=0;

	wmemb = fwrite(ptr, size, nmemb, stream);
	if (wmemb < nmemb) {
		fprintf(stderr, "Error while writing to file!\n");
		exit(EXIT_FAILURE);
	}
	return wmemb;
}

size_t Fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {

	size_t rmemb=0;

	rmemb = fread(ptr, size, nmemb, stream);
	if (rmemb < nmemb && ferror(stream)) {
		fprintf(stderr, "Error while reading from file, could not read more than %zu elements!\n", rmemb);
		exit(EXIT_FAILURE);
	}
	return rmemb;
}

int Remove(const char *pathname) {

	int rv = remove(pathname);
	if (rv && errno != ENOENT) {
		fprintf(stderr, "Error removing file %s: %s\n", pathname, strerror(errno));
		exit(EXIT_FAILURE);
	}
	return rv;
}

off_t getFsize(const char *fpath) {

        struct stat     st;
        int             rv;

        rv = stat(fpath, &st);
        if (rv) {
                fprintf(stderr, "Cannot stat file %s...\n", fpath);
                exit(EXIT_FAILURE);
        }
        return st.st_size;
}

double Wtime(void) {
#ifdef _MSC_VER
	static LARGE_INTEGER frequency;
	if (frequency.QuadPart == 0)
		::QueryPerformanceFrequency(&frequency);
	LARGE_INTEGER now;
	::QueryPerformanceCounter(&now);
	return now.QuadPart / double(frequency.QuadPart);
#else
	struct timespec tp;
	int rv = clock_gettime(CLOCK_MONOTONIC, &tp);
	if (rv) return 0;
	return tp.tv_nsec / 1.0E+9 + (double)tp.tv_sec;
#endif

}

