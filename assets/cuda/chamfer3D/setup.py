from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer3D',
    ext_modules=[
        CUDAExtension('chamfer3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer3D_cuda.cpp']), # must named as xxx_cuda.cpp
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='1.0.1')
