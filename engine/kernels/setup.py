from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="quant_engine",
    ext_modules=[
        CUDAExtension(
            name="quant_engine",
            sources=[
                "pybind.cpp",
                "tpack/tpack.cu",
                "functions/linear.cu",
                "functions/quantlinear.cu",
                "functions/quantlinear_float_input.cu",
                "functions/conv2d.cu",
                "functions/quantconv2d.cu",
                "functions/quantconv2d_float_input.cu",
            ]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
