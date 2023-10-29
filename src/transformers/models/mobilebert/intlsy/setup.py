from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mobilebert_cpp',
    ext_modules=[
        CUDAExtension('mobilebert_cpp', [
            'mobilebert_cpp_kernels.cu',
            'mobilebert_cpp.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
