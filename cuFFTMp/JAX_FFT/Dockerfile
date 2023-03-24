FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3

RUN python3.8 -m pip install --upgrade "jax[cuda]==0.4.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

COPY . /fft_jax
RUN rm -rf /fft_jax/build
RUN pip install -e /fft_jax

ENV LD_LIBRARY_PATH=/fft_jax/nvshmem/lib:/fft_jax/cufftmp/lib:$LD_LIBRARY_PATH

ENV NVSHMEM_DISABLE_NCCL=1
ENV NVSHMEM_DISABLE_GDRCOPY=1
ENV NVSHMEM_BOOTSTRAP=MPI

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
