export CURR_PATH=`pwd`
cd ${CURR_PATH} && cd bench_sddmm_csr && make -j && ./bench_sddmm_csr
cd ${CURR_PATH} && cd bench_sparse2dense_csr && make -j && ./bench_sparse2dense_csr
cd ${CURR_PATH} && cd bench_spmm_coo && make -j && ./bench_spmm_coo
cd ${CURR_PATH} && cd bench_spmm_csr && make -j && ./bench_spmm_csr
cd ${CURR_PATH} && cd bench_spmm_csr_op && make -j && ./bench_spmm_csr_op
