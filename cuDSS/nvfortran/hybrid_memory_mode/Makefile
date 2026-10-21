# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone Makefile for the cuDSS CUDA Fortran example "hybrid_memory_mode" (nvfortran).
# Set the variables below to point at your cuDSS and CUDA installation.

EXAMPLE   := hybrid_memory_mode

# GNU make predefines FC=f77 (origin "default"), which ?= would not override, so
# default it explicitly while still honoring an FC given on the command line or in
# the environment.
ifeq ($(origin FC),default)
FC := nvfortran
endif
FFLAGS    ?= -O2 -cuda
CUDSS_INC ?= cudss_inc
CUDSS_LIB ?= cudss_lib
CUDSS_INTERFACE_DIR ?= $(CUDSS_INC)/fortran

MODFLAG := -module .
LIBS    := $(CUDSS_LIB)/libcudss.so -cudalib=cublas -Wl,-rpath,$(CUDSS_LIB)
FFLAGS  += -I.

$(EXAMPLE): cudss.o validation.o cudssutils.o $(EXAMPLE).o
	$(FC) $(FFLAGS) $(LDFLAGS) $(MODFLAG) $^ $(LIBS) -o $@

cudss.o: $(CUDSS_INTERFACE_DIR)/cudss.f90
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
validation.o: validation.f90
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
cudssutils.o: cudssutils.cuf cudss.o
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@
$(EXAMPLE).o: $(EXAMPLE).cuf cudss.o cudssutils.o validation.o
	$(FC) $(FFLAGS) $(MODFLAG) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o *.mod $(EXAMPLE)
