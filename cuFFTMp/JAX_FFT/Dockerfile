# We use JAX-toolbox from https://github.com/NVIDIA/JAX-Toolbox
FROM ghcr.io/nvidia/jax:jax-2024-10-24
RUN apt-get update && apt-get install openmpi-bin -y

COPY . /fft_jax
RUN rm -rf /fft_jax/build
RUN pip install -e /fft_jax

ENV LD_LIBRARY_PATH=/fft_jax/nvshmem/lib:/fft_jax/cufftmp/lib:$LD_LIBRARY_PATH

ENV NVSHMEM_DISABLE_NCCL=1
ENV NVSHMEM_DISABLE_GDRCOPY=1
ENV NVSHMEM_BOOTSTRAP=MPI

# Infiniband service level is beneficial for performance for large FFTs on many GPUs.
# see *Note* in https://docs.nvidia.com/hpc-sdk/cufftmp/usage/performances.html#performance-considerations
# The IB service level for both NVSHMEM (for cuFFTMp) and NCCL (for JAX FFT) are declared here.
ENV NVSHMEM_IB_SL=1
ENV NCCL_IB_SL=1

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1