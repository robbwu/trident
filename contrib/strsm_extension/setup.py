from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Point to your CUDA installation if necessary
# cuda_home = '/usr/local/cuda'
# os.environ['CUDA_HOME'] = cuda_home

setup(
    name='strsm_extension',
    ext_modules=[
        CUDAExtension(
            name='strsm_extension', # Must match the PYBIND11_MODULE name argument
            sources=['strsm_extension.cu'],
            libraries=['cublas'] # Link against the cuBLAS library
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
