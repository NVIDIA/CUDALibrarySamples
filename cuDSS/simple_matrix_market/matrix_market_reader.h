#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

template <typename _TYPE_>
int matrix_reader(std::string filename, int& n, int& nnz, int** csr_offsets_h,
                   int** csr_columns_h, _TYPE_** csr_values_h, cudssMatrixViewType_t mview) {
    
    std::ifstream file(filename);  // Open file for reading

    if (!file.is_open()) {
        std::cerr << "\033[38;5;196m Error: Could not open file \033[0m" << filename << std::endl;
        return EXIT_FAILURE;
    }

    std::string line;
    int i_val = 0;
    int i_offsets = 0;
    int cumulative_nnz = 0;
    int previous_line = -1;
    bool foundSize = false;

    bool found_lower = false;
    bool found_upper = false;

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

            std::cout << "MATRIX READER: allocated host memory\n";

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

            if (i>j)
                found_lower = true;
            if (i<j)
                found_upper = true;

        }
    }

    if ((found_lower)&&(mview == CUDSS_MVIEW_UPPER)){
        std::cerr<<"\033[38;5;196m ERROR: mview specified is upper, but reader found an element in the lower triangular part of the matrix \033[0m"<<std::endl;
        return EXIT_FAILURE;
    }

    if ((found_upper)&&(mview == CUDSS_MVIEW_LOWER)){
        std::cerr<<"\033[38;5;196m ERROR: mview specified is lower, but reader found an element in the upper triangular part of the matrix \033[0m"<<std::endl;
        return EXIT_FAILURE;
    }

    if(not((found_upper)&&(found_lower))&&(mview == CUDSS_MVIEW_FULL)){
        std::cerr<<"\033[38;5;196m ERROR: mview specified is full, but reader found only lower OR upper elements \033[0m"<<std::endl;
        return EXIT_FAILURE;   
    }

    (*csr_offsets_h)[i_offsets++] = cumulative_nnz;
    if (cumulative_nnz != nnz) {
        std::cerr << "\033[38;5;196m READER: Error: nnz != cumulative_nnz \033[0m" << std::endl;
        return EXIT_FAILURE;
    }
    file.close();  // Optional; automatic when `file` goes out of scope

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