include ../common.mk 

exe = cufftmp_c2c
exe_acc = cufftmp_c2c_acc

all : $(exe) $(exe_acc) 

$(exe) : $(exe).o   
	$(f90) -o $@ $+ $(FLAGS) $(LINKER)

$(exe_acc) : $(exe_acc).o   
	$(f90) -o $@ $+ $(FLAGS) $(LINKER) -acc 

%.o : %.f90
	$(f90) -c $< -o $@ $(FLAGS)

%_acc.o : %.f90
	$(f90) -c $< -o $@ $(FLAGS) -DACC -acc 

.PHONY: clean

clean:
	@rm -rf *.mod *.o  $(exe) $(exe_acc)

run: $(exe) $(exe_acc)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -n 2 $(exe)
	LD_LIBRARY_PATH="${NVSHMEM_LIB}:${CUFFT_LIB}:${LD_LIBRARY_PATH}" mpirun -n 2 $(exe_acc) 
