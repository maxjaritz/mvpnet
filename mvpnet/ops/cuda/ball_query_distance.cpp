#ifndef _BALL_QUERY
#define _BALL_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> BallQueryDistance(
    const at::Tensor query,
    const at::Tensor key,
    const float radius,
    const int64_t max_neighbors);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query_distance", &BallQueryDistance, "Ball query with distance (CUDA)");
}

#endif