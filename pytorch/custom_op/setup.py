from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='prop_tcfg_cuda',
    ext_modules=[
        CUDAExtension('prop_tcfg_cuda', [
            'prop_tcfg_cuda.cpp',
            'prop_tcfg_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

