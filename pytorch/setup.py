import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='mpelu_cuda',
    version='1.6',
    ext_modules=[
        CUDAExtension(
            name='mpelu_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3'],
                # Keep line information for profiling without enabling CUDA's
                # device-debug mode (-G), which disables key optimizations.
                'nvcc': ['-O3', '-lineinfo']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
