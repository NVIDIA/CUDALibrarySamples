export CURR_PATH=`pwd`
cd ${CURR_PATH} && cd bench_sddmm_csr && make -j && ./bench_sddmm_csr--A_num_rows=100 --A_num_cols=100 --B_num_cols=100 --C_sparsity=0.5
cd ${CURR_PATH} && cd bench_sparse2dense_csr && make -j && ./bench_sparse2dense_csr --num_rows=100 --num_cols=100 --sparsity=0.5
cd ${CURR_PATH} && cd bench_spmm_coo && make -j && ./bench_spmm_coo --A_num_rows=100 --A_num_cols=100 --B_num_cols=100 --A_sparsity=0.5
cd ${CURR_PATH} && cd bench_spmm_csr && make -j && ./bench_spmm_csr --A_num_rows=100 --A_num_cols=100 --B_num_cols=100 --A_sparsity=0.5
cd ${CURR_PATH} && cd bench_spmm_csr_op && make -j && ./bench_spmm_csr_op --A_num_rows=100 --A_num_cols=100 --B_num_cols=100 --A_sparsity=0.5
