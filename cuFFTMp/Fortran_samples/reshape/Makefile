include ../common.mk 

exe = cufftmp_reshape

all : $(exe)  

$(exe) : $(exe).o   
	$(f90) -o $@ $+ $(FLAGS) $(LINKER) 

%.o : %.f90
	$(f90) -c $< -o $@ $(FLAGS)

.PHONY: clean

clean:
	@rm -rf *.mod *.o  $(exe) 

run: $(exe) 
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -n 2 $(exe)
