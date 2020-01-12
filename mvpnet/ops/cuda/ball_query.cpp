#ifndef _BALL_QUERY
#define _BALL_QUERY

#include <vector>
#include <torch/extension.h>

at::Tensor BallQuery(
    const at::Tensor query,
    const at::Tensor key,
    const float radius,
    const int64_t max_neighbors);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &BallQuery, "Ball query (CUDA)");
}

#endif