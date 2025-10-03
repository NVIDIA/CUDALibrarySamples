#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

enum MtxReaderStatus {
    MtxReaderSuccess,
    MtxReaderErrorFileNotFound,
    MtxReaderErrorFileMemAllocFailed,
    MtxReaderErrorWrongNnz,
    MtxReaderErrorUpperViewButLowerFound,
    MtxReaderErrorLowerViewButUpperFound,
    MtxReaderErrorOutOfBoundRowIndex,
    MtxReaderErrorOfBoundColIndex

};

template <typename _TYPE_>
int matrix_reader(std::string filename, int& n, int& nnz, int** csr_offsets_h,
                  int** csr_columns_h, _TYPE_** csr_values_h, cudssMatrixViewType_t mview, bool verbose=true, bool verbose_err = true)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        if (verbose_err)
            std::cerr << "\033[38;5;196m Error: Could not open file \033[0m" << filename << std::endl;
        // ! Do a mock malloc anyway so that subsequent free does not fail !
        *csr_offsets_h = (int*)malloc(1);
        *csr_columns_h = (int*)malloc(1);
        *csr_values_h = (_TYPE_*)malloc(1);
        return  MtxReaderErrorFileNotFound;
    }

    std::string line;
    bool foundSize = false;
    int declared_nnz = 0;
    int file_line_count = 0;

    //Tuple for the mtw lines
    std::vector<std::tuple<int, int, _TYPE_>> entries;

    bool found_lower = false, found_upper = false;

    while (std::getline(file, line)) {
        //Skip header
        if (line.empty() || line[0] == '%') continue;

        //Get metadata
        std::istringstream lineData(line);
        if (!foundSize) {
            int n1;
            lineData >> n >> n1 >> declared_nnz;
            if (verbose)
                std::cout << "MATRIX READER: n0= " << n << ", n1= " << n1 << ", nnz= " << declared_nnz << "\n";
            foundSize = true;
        //Get line
        } else {
            int i, j;
            _TYPE_ val;
            lineData >> i >> j >> val;
            i -= 1; j -= 1;  // Convert from 1-based to 0-based

            //Append the tuple to entries (resize everytime so sub-optimal)
            entries.emplace_back(i, j, val);

            if (i > j) found_lower = true;
            if (i < j) found_upper = true;
            file_line_count+=1;
        }
    }

    file.close();
    nnz = entries.size();

    // Allocate memory
    *csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    *csr_columns_h = (int*)malloc(nnz * sizeof(int));
    *csr_values_h = (_TYPE_*)malloc(nnz * sizeof(_TYPE_));

    if (!(*csr_offsets_h) || !(*csr_columns_h) || !(*csr_values_h)) {
        if (verbose_err)
            std::cerr << "\033[38;5;196m ERROR: Memory allocation failed \033[0m\n";
        return MtxReaderErrorFileMemAllocFailed;
    }

    if (file_line_count != declared_nnz) {
        if (verbose_err)
            std::cerr << "\033[38;5;196m ERROR: more elements in the mtx file than announced in the header\033[0m\n";
        return MtxReaderErrorWrongNnz;
    }


    // Sort by row, then by column, 3rd argument is the compare operator, (row then column)
    std::sort(entries.begin(), entries.end(), [](auto& a, auto& b) {
        if (std::get<0>(a) == std::get<0>(b))
            return std::get<1>(a) < std::get<1>(b);
        return std::get<0>(a) < std::get<0>(b);
    });


    //Some checks
    if ((found_lower) && (mview == CUDSS_MVIEW_UPPER)) {
        if (verbose_err)
            std::cerr << "\033[38;5;196m ERROR: mview is upper, but lower elements found \033[0m\n";
        return MtxReaderErrorUpperViewButLowerFound;
    }
    if ((found_upper) && (mview == CUDSS_MVIEW_LOWER)) {
        if (verbose_err)
            std::cerr << "\033[38;5;196m ERROR: mview is lower, but upper elements found \033[0m\n";
        return MtxReaderErrorLowerViewButUpperFound;
    }
    if (!(found_upper && found_lower) && mview == CUDSS_MVIEW_FULL) {
        if (verbose)
            std::cout << "\033[38;5;208m WARNING: mview is full, but only lower or upper elements found \033[0m\n";
    }

    //Initialize with 0's
    std::fill(*csr_offsets_h, *csr_offsets_h + (n + 1), 0);

    // First pass: count entries per row
    for (auto& [i, j, val] : entries) {
        if (i >= n || i < 0) {
            if (verbose_err)
                std::cerr << "\033[38;5;196m ERROR: Invalid row index " << i << " \033[0m\n";
            return MtxReaderErrorOutOfBoundRowIndex;
        }
        if (j >= n || j < 0) {
            if (verbose_err)
                std::cerr << "\033[38;5;196m ERROR: Invalid col index " <<j << " \033[0m\n";
            return MtxReaderErrorOfBoundColIndex;
        }
        (*csr_offsets_h)[i + 1]++;
    }

    // Prefix sum for empty rows
    for (int i = 0; i < n; ++i) {
        (*csr_offsets_h)[i + 1] += (*csr_offsets_h)[i];
    }

    // Second pass: fill in column and value arrays
    std::vector<int> row_fill(n, 0);
    for (auto& [i, j, val] : entries) {
        int offset = (*csr_offsets_h)[i] + row_fill[i];
        (*csr_columns_h)[offset] = j;
        (*csr_values_h)[offset] = val;
        row_fill[i]++;
    }

    // // Detect empty rows
    for (int i = 0; i < n; ++i) {
        if ((*csr_offsets_h)[i] == (*csr_offsets_h)[i + 1]) {
            if (verbose)
                std::cout << "\033[38;5;208m Warning: Row " << i << " is empty \033[0m\n";
        }
    }

    if (verbose)
        std::cout << "MATRIX READER: Completed with " << nnz << " nonzeros\n";
    return 0;
}

template <typename _TYPE_>
int rhs_reader(std::string filename, int& n, _TYPE_** b_values_h) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "\033[38;5;196m Error: Could not open file \033[0m" << filename << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    bool foundSize = false;

    int row = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream lineData(line);

        if (!foundSize) {
            int dummy0, dummy1;
            lineData >> n >> dummy0 >> dummy1;
            *b_values_h = (_TYPE_*)malloc(n * sizeof(_TYPE_));
            std::cout << "VECTOR READER: allocated host memory\n";
            std::cout << "VECTOR READER: n = " << n << "\n";
            foundSize = true;
        } else {
            _TYPE_ val;
            lineData >> val;
            (*b_values_h)[row] = val;  
            row += 1;
        }
    }

    file.close();
    return 0;
}