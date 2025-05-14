#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

template <typename _TYPE_>
void matrix_reader(std::string filename, int& n, int& nnz, int** csr_offsets_h,
                   int** csr_columns_h, _TYPE_** csr_values_h) {
    
    std::ifstream file(filename);  // Open file for reading

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }

    std::string line;
    int i_val = 0;
    int i_offsets = 0;
    int cumulative_nnz = 0;
    int previous_line = -1;
    bool foundSize = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream lineData(line);

        if (!foundSize) {
            int n1;
            lineData >> n >> n1 >> nnz;
            std::cout << "MATRIX READER: n0= " << n << ", n1= " << n1 << ", nnz= " << nnz << "\n";

            *csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
            *csr_columns_h = (int*)malloc(nnz * sizeof(int));
            *csr_values_h = (_TYPE_*)malloc(nnz * sizeof(_TYPE_));
            std::cout << "MATRIX READER: allocated memory\n";

            foundSize = true;
        } else {
            int i, j;
            _TYPE_ val;
            lineData >> i >> j >> val;

            if (i > previous_line) {
                (*csr_offsets_h)[i_offsets++] = cumulative_nnz;
                previous_line = i;
            }

            (*csr_columns_h)[i_val] =
                j - 1;  //-1 because mtx indexing is fortran like
            (*csr_values_h)[i_val++] = val;
            cumulative_nnz += 1;
        }
    }

    (*csr_offsets_h)[i_offsets++] = cumulative_nnz;
    if (cumulative_nnz != nnz) {
        std::cerr << "READER: Error: nnz != cumulative_nnz" << std::endl;
    }
    file.close();  // Optional; automatic when `file` goes out of scope
}

template <typename _TYPE_>
void rhs_reader(std::string filename, int& n, _TYPE_** b_values_h) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    bool foundSize = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;

        std::istringstream lineData(line);

        if (!foundSize) {
            int dummy0, dummy1;
            lineData >> n >> dummy0 >> dummy1;
            *b_values_h = (_TYPE_*)malloc(n * sizeof(_TYPE_));
            std::cout << "VECTOR READER: n = " << n << "\n";
            foundSize = true;
        } else {
            int row, col;
            _TYPE_ val;
            lineData >> row >> col >> val;
            (*b_values_h)[row - 1] =
                val;  //-1 because mtx indexing is fortran like
        }
    }

    file.close();
}