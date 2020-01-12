/* CUDA Implementation for ball xyz2*/
#ifndef _BALL_QUERY_KERNEL
#define _BALL_QUERY_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>  // at::cuda::getApplyGrid
#include <THC/THC.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// NOTE: AT_CHECK has become TORCH_CHECK on master after 1.2.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define MAX_THREADS 512

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return max(min(1 << pow_2, MAX_THREADS), 1);
}

// From getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
inline bool getGrid(uint64_t numBlocks, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

#define RUN(BLOCK_SIZE) \
  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "BallQueryForward", ([&] { \
    BallQueryForwardKernel<BLOCK_SIZE, 3, scalar_t, int64_t> \
      <<<grid, BLOCK_SIZE>>>( \
      index.data<int64_t>(), \
      query.data<scalar_t>(), \
      key.data<scalar_t>(), \
      batch_size, \
      n1, \
      n2, \
      (scalar_t)radius, \
      max_neighbors); \
  }));

#define RUN_BLOCK(BLOCK_SIZE) \
  case BLOCK_SIZE: \
    RUN(BLOCK_SIZE) \
    break;

/*
Forward kernel
Load a block of key data and process a block of query data
*/
template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void BallQueryForwardKernel(
    index_t* __restrict__ index,
    const scalar_t *__restrict__ query,
    const scalar_t *__restrict__ key,
    const int64_t batch_size,
    const int64_t n1,
    const int64_t n2,
    const scalar_t radius,
    const int64_t max_neighbors) {

  // calculate the number of blocks
  const int num_block1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int num_block2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int total_blocks = batch_size * num_block1;
  const scalar_t radius_square = radius * radius;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x) {
    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM];
    const int batch_idx = block_idx / num_block1;
    const int block_idx1 = block_idx % num_block1;
    const int query_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int query_offset = (batch_idx * n1 + query_idx) * DIM;
    
    // load current query point
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < n1) {
      #pragma unroll
      for (int i = 0; i < DIM; ++i) {
        cur_query[i] = query[query_offset + i];
      }
    }
    
    index_t cnt_neighbors = 0;
    const int index_offset = batch_idx * n1 * max_neighbors + query_idx * max_neighbors;
    // load a block of key data to reduce the time to read data
    for (int block_idx2 = 0; block_idx2 < num_block2; ++block_idx2) {
      // load key data
      int key_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int key_offset = (batch_idx * n2 + key_idx) * DIM;
      if (key_idx < n2) {
        #pragma unroll
        for (int i = 0; i < DIM; ++i) {
          key_buffer[threadIdx.x * DIM + i] = key[key_offset + i];
        }
      }
      __syncthreads();
      
      // calculate the distance between current query and key, with the shared memory.
      if (query_idx < n1) {
        for (int j = 0; j < BLOCK_SIZE; ++j) {
          int key_idx2 = (block_idx2 * BLOCK_SIZE) + j;
          const int buffer_offset = j * DIM;
          scalar_t dist = 0.0;
          #pragma unroll
          for (int i = 0; i < DIM; ++i) {
            scalar_t diff = key_buffer[buffer_offset + i] - cur_query[i];
            dist += diff * diff;
          }
          if (key_idx2 < n2 && cnt_neighbors < max_neighbors) {
            if (dist < radius_square) {
              index[index_offset + cnt_neighbors] = key_idx2;
              ++cnt_neighbors;
            }
          }
        }
      }
      __syncthreads();
    }
    // pad with the first term if necessary
    if (query_idx < n1 && cnt_neighbors < max_neighbors) {
      index_t pad_val = index[index_offset];
      for (int j = cnt_neighbors; j < max_neighbors; ++j) {
        index[index_offset + j] = pad_val;
      }
    }
  }
}

/*
Forward interface
Input:
  query: (B, N1, 3)
  key: (B, N2, 3)
  radius: float
  max_neighbors: int
Output:
  index: (B, N1, K)
*/
at::Tensor BallQuery(
    const at::Tensor query,
    const at::Tensor key,
    const float radius,
    const int64_t max_neighbors) {

  const auto batch_size = query.size(0);
  const auto n1 = query.size(1);
  const auto n2 = key.size(1);

  // Sanity check
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_EQ(query.size(2), 3);
  CHECK_EQ(key.size(2), 3);

  // Allocate new space for output
  auto index = at::full({batch_size, n1, max_neighbors}, -1, query.type().toScalarType(at::kLong));
  index.set_requires_grad(false);
  
  // Calculate grids and blocks for kernels
  const auto n_threads = opt_n_threads(min(n1, n2));
  const auto num_blocks1 = (n1 + n_threads - 1) / n_threads;
  dim3 grid;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * num_blocks1, grid, curDevice);
  
  switch (n_threads) {
    RUN_BLOCK(512)
    RUN_BLOCK(256)
    RUN_BLOCK(128)
    RUN_BLOCK(64)
    RUN_BLOCK(32)
    default:
      RUN(16)
  }

  THCudaCheck(cudaGetLastError());

  return index;
}

#endif