#ifndef _INTERPOLATE
#define _INTERPOLATE

#include <vector>
#include <torch/extension.h>

//CUDA declarations
at::Tensor InterpolateForward(
    const at::Tensor input,
    const at::Tensor index,
    const at::Tensor weight);

at::Tensor InterpolateBackward(
    const at::Tensor grad_output,
    const at::Tensor index,
    const at::Tensor weight,
    const int64_t num_inst);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("interpolate_forward", &InterpolateForward,"Interpolate feature forward (CUDA)");
  m.def("interpolate_backward", &InterpolateBackward, "Interpolate feature backward (CUDA)");
}

#endif
