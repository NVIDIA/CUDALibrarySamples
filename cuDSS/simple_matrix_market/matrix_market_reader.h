/* 
SPDX-FileCopyrightText: Copyright (c) 2025 Rémi Bourgeois. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 
*/

#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <tuple>

enum MtxReaderStatus {
    MtxReaderSuccess,
    MtxReaderErrorFileNotFound,
    MtxReaderErrorFileMemAllocFailed,
    MtxReaderErrorWrongNnz,
    MtxReaderErrorUpperViewButLowerFound,
    MtxReaderErrorLowerViewButUpperFound,
    MtxReaderErrorOutOfBoundRowIndex,
    MtxReaderErrorOfBoundColIndex,
    MtxReaderErrorInvalidFormatInHeader
};

template <typename value_type>
int matrix_reader(std::string filename, int& n, int& nnz, int** csr_offsets_h,
                  int** csr_columns_h, value_type** csr_values_h, cudssMatrixViewType_t mview)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open file %s\n", filename.c_str());
        *csr_offsets_h = NULL;
        *csr_columns_h = NULL;
        *csr_values_h = NULL;
        return  MtxReaderErrorFileNotFound;
    }

    std::string line;
    bool foundSize = false;
    int declared_nnz = 0;
    int file_line_count = 0;
    int foundHeader = false;

    //Tuple for the mtw lines
    std::vector<std::tuple<int, int, value_type>> entries;

    bool FoundAlowerElement = false, FoundAUpperElement = false;

    while (std::getline(file, line)) {
        
        //Get line
        std::istringstream lineData(line);
        
        //skip empty lines
        if (line.empty()) 
            continue;

        //skip comments
        if (foundHeader && line[0]=='%') 
            continue;

        //Get header
        if (!foundHeader && line.substr(0, 14) == "%%MatrixMarket"){
            foundHeader = true;
                
            // Parse header: %%MatrixMarket matrix coordinate real general
            std::string marker, object, format, value_type_header, symmetry;
            
            lineData >> marker >> object >> format >> value_type_header >> symmetry;
            
            // Check if it's matrix coordinate real
            if (object != "matrix" || format != "coordinate" || value_type_header != "real") {
                fprintf(stderr, "ERROR: Expected 'matrix coordinate real for the matrix file header\n");
                fprintf(stderr, "Got: object=%s format=%s value_type=%s\n", 
                        object.c_str(), format.c_str(), value_type_header.c_str());
                    return MtxReaderErrorInvalidFormatInHeader;
            }
            continue;
        }

        //Get sizes
        if (!foundSize) {
            int n1;
            lineData >> n >> n1 >> declared_nnz;
            printf("MATRIX READER: n0= %d, n1= %d, nnz= %d\n", n, n1, declared_nnz);
            foundSize = true;
        //Get line
        } else {
            int i, j;
            value_type val;
            lineData >> i >> j >> val;
            i -= 1; j -= 1;  // Convert from 1-based to 0-based

            //Append the tuple to entries (resize everytime so sub-optimal)
            entries.emplace_back(i, j, val);

            if (i > j) FoundAlowerElement = true;
            if (i < j) FoundAUpperElement = true;
            file_line_count+=1;
        }
    }

    file.close();
    nnz = entries.size();

    // Allocate memory
    *csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    *csr_columns_h = (int*)malloc(nnz * sizeof(int));
    *csr_values_h = (value_type*)malloc(nnz * sizeof(value_type));

    if (!(*csr_offsets_h) || !(*csr_columns_h) || !(*csr_values_h)) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        return MtxReaderErrorFileMemAllocFailed;
    }

    if (file_line_count != declared_nnz) {
        fprintf(stderr, "ERROR: more elements in the mtx file than announced in the header\n");
        return MtxReaderErrorWrongNnz;
    }

    std::sort(entries.begin(), entries.end());

    if ((FoundAlowerElement) && (mview == CUDSS_MVIEW_UPPER)) {
        fprintf(stderr, "ERROR: mview is upper, but lower elements found\n");
        return MtxReaderErrorUpperViewButLowerFound;
    }

    if ((FoundAUpperElement) && (mview == CUDSS_MVIEW_LOWER)) {
        fprintf(stderr, "ERROR: mview is lower, but upper elements found\n");
        return MtxReaderErrorLowerViewButUpperFound;
    }

    if (!(FoundAUpperElement && FoundAlowerElement) && mview == CUDSS_MVIEW_FULL) {
        fprintf(stdout, "WARNING: mview is full, but only lower or upper elements found\n");
    }

    //Initialize with 0's
    std::fill(*csr_offsets_h, *csr_offsets_h + (n + 1), 0);

    int current_idx = 0;

    for (auto& [i, j, val] : entries) {
        if (i >= n || i < 0) {
            fprintf(stderr, "ERROR: Invalid row index %d\n", i);
            return MtxReaderErrorOutOfBoundRowIndex;
        }
        if (j >= n || j < 0) {
            fprintf(stderr, "ERROR: Invalid col index %d\n", j);
            return MtxReaderErrorOfBoundColIndex;
        }
        (*csr_offsets_h)[i + 1]++;
        (*csr_columns_h)[current_idx] = j;
        (*csr_values_h)[current_idx] = val;
        current_idx++;
    }

    // Prefix sum to compute the offsets
    for (int i = 0; i < n; ++i) 
        (*csr_offsets_h)[i + 1] += (*csr_offsets_h)[i];
        

    // Detect empty rows
    for (int i = 0; i < n; ++i) {
        if ((*csr_offsets_h)[i] == (*csr_offsets_h)[i + 1]) {
            printf("Warning: Row %d is empty\n", i);
        }
    }

    printf("MATRIX READER: Completed with %d nonzeros\n", nnz);
    return 0;
}

template <typename value_type>
int rhs_reader(std::string filename, int& n, value_type** b_values_h) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open file %s\n", filename.c_str());
        return EXIT_FAILURE;
    }

    std::string line;
    bool foundSize = false;
    bool foundHeader = false;
    int row = 0;
    while (std::getline(file, line)) {
        //Get line
        std::istringstream lineData(line);
        
        //skip empty lines
        if (line.empty()) 
            continue;

        //skip comments
        if (foundHeader && line[0]=='%') 
            continue;

        //Get header
        if (!foundHeader && line.substr(0, 14) == "%%MatrixMarket"){
            foundHeader = true;
                
            // Parse header: %%MatrixMarket matrix coordinate real general
            std::string marker, object, format, value_type_header, symmetry;
            
            lineData >> marker >> object >> format >> value_type_header >> symmetry;
            
            // Check if it's matrix coordinate real
            if (object != "matrix" || format != "array" || value_type_header != "real") {
                fprintf(stderr, "ERROR: Expected 'matrix array real for the rhs file header\n");
                fprintf(stderr, "Got: object=%s format=%s value_type=%s\n", 
                        object.c_str(), format.c_str(), value_type_header.c_str());
                    return MtxReaderErrorInvalidFormatInHeader;
            }
            continue;
        }

        if (!foundSize) {
            int num_rhs;
            lineData >> n >> num_rhs;
            if (num_rhs != 1) {
                fprintf(stderr, "Only one RHS is supported, %d found\n.", num_rhs);
                return EXIT_FAILURE;
            }
            *b_values_h = (value_type*)malloc(n * sizeof(value_type));
            printf("RHS READER: allocated host memory\n");
            printf("RHS READER: n = %d\n", n);
            foundSize = true;
        } else {
            value_type val;
            lineData >> val;
            (*b_values_h)[row] = val;  
            row += 1;
        }
    }

    file.close();
    return 0;
}