from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = ['-O3', '-arch=native', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
nvcc_flags.append('--use_fast_math')

setup(
    name='mobilebert_cpp',
    ext_modules=[
        CUDAExtension('mobilebert_cpp',
			[
                'mobilebert_cpp_kernels.cu',
                'mobilebert_cpp.cpp',
            ],
            extra_compile_args={
                'gcc': [],
                'nvcc': nvcc_flags
            }
		)
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
