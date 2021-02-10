# Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

FLAGS=INC=-I$(GPGPU)/cusparse/branches/dev/include\                            \
          -I$(GPGPU)/cuda/tools/cudart\                                        \
          -I$(GPGPU)/bin/x86_64_Linux_release\                                 \
          -I$(GPGPU)/bin/x86_64_Linux_release\include

all:
	make -C spvv            $(FLAGS)
	make -C axpby	        $(FLAGS)
	make -C gather          $(FLAGS)
	make -C rot             $(FLAGS)
	make -C scatter         $(FLAGS)
	make -C sparse2dense    $(FLAGS)
	make -C dense2sparse    $(FLAGS)
	make -C spmv_csr        $(FLAGS)
	make -C spmv_coo        $(FLAGS)
	make -C spsv_csr        $(FLAGS)
	make -C spmm_csr        $(FLAGS)
	make -C spmm_coo        $(FLAGS)
	make -C spmm_blockedell $(FLAGS)
	make -C sddmm_csr       $(FLAGS)
	make -C spgemm          $(FLAGS)
	make -C compression     $(FLAGS)
	make -C graph_capture   $(FLAGS)

clean:
	rm -f spvv/spvv_example                                                    \
          axpby/axpby_example                                                  \
          gather/gather_example                                                \
          rot/rot_example                                                      \
          scatter/scatter_example                                              \
          sparse2dense/sparse2dense_example                                    \
          dense2sparse/dense2sparse_example                                    \
          spmv_csr/spmv_csr_example                                            \
          spmv_coo/spmv_coo_example                                            \
          spsv_csr/spsv_csr_example                                            \
          spmm_csr/spmm_csr_example                                            \
          spmm_coo/spmm_coo_example                                            \
          spmm_blockedell/spmm_blockedell_example                              \
          sddmm_csr/sddmm_csr_example                                          \
          spgemm/spgemm_example                                                \
          compression/compression_example                                      \
          graph_capture/graph_capture_example

test:
	spvv/spvv_example
	axpby/axpby_example
	gather/gather_example
	rot/rot_example
	scatter/scatter_example
	sparse2dense/sparse2dense_example
	dense2sparse/dense2sparse_example
	spmv_csr/spmv_csr_example
	spmv_coo/spmv_coo_example
	spsv_csr/spsv_csr_example
	spmm_csr/spmm_csr_example
	spmm_coo/spmm_coo_example
	spmm_blockedell/spmm_blockedell_example
	sddmm_csr/sddmm_csr_example
	spgemm/spgemm_example
	compression/compression_example
	graph_capture/graph_capture_example
