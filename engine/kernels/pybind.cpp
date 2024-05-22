#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "tpack/tpack.h"
#include "functions/funcs.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tpack", &tpack, "Packs the given tensor into a vector of tensors.");
    m.def("tunpack", &tunpack, "Unpacks the given vector of tensors into a tensor.");
    m.def("linear", &linear, "Linear function.");
    m.def("quantlinear", &quantlinear, "Quantized linear function.");
    m.def("quantlinear_float_input", &quantlinear_float_input, "Quantized linear function with float input.");
    m.def("conv2d", &conv2d, "Conv2d function.");
    m.def("quantconv2d", &quantconv2d, "Quantized conv2d function.");
    m.def("quantconv2d_float_input", &quantconv2d_float_input, "Quantized conv2d function with float input.");
}
