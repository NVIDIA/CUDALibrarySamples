#!/usr/bin/env python

import codecs
import os
import subprocess
import sys
import distutils.sysconfig
import pybind11

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))

def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

class CMakeBuildExt(build_ext):
    def build_extensions(self):

        cmake_python_library = "{}/{}".format(
            distutils.sysconfig.get_config_var("LIBDIR"),
            distutils.sysconfig.get_config_var("INSTSONAME"),
        )
        cmake_python_include_dir = distutils.sysconfig.get_python_inc()

        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy"))
        )
        os.makedirs(install_dir, exist_ok=True)
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_CUDA_ARCHITECTURES=70;80;90",
            "-DNVSHMEM_HOME={}".format(HERE + "/nvshmem"),
            "-DCUFFTMP_HOME={}".format(HERE + "/cufftmp"),
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release"
            ),
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
        ]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", f"{HERE}/src/cufftmp_jax/"] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext):
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )


extensions = [
    Extension(
        "cufftmp_jax.gpu_ops",
        [
            "src/cufftmp_jax/lib/gpu_ops.cpp",
            "src/cufftmp_jax/lib/kernels.cu",
        ],
    )]


setup(
    name="fft_jax",
    version='0.0.1',
    author="Leopold Cambier",
    author_email="lcambier@nvidia.com",
    license="All rights reserved",
    description=("FFT + JAX"),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["jax[cuda]", "jaxlib"],
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
