#include "mobilebert_cpp_kernels.h"

#include <stdexcept>
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

static constexpr int64_t BATCH_PER_BLOCK = 16;

// Block shape: [hidden_size/2]
template<typename T>
__global__ void fused_add_mult_add_kernel(
	T* output,					// [batch_size, hidden_size]
	const T* __restrict__ a,	// [batch_size, hidden_size]
	const T* __restrict__ b,	// [batch_size, hidden_size]
	const T* __restrict__ c,	// [hidden_size]
	const T* __restrict__ d,	// [hidden_size]
	int64_t batch_size,
	int64_t hidden_size
) {
	if constexpr(std::is_same<T, half>::value) {
		typedef half2 T2;
		T2 my_c = ((const T2*)c)[threadIdx.x];
		T2 my_d = ((const T2*)d)[threadIdx.x];

		#pragma unroll 4
		for (int64_t i = 0; i < BATCH_PER_BLOCK && i + blockIdx.x*BATCH_PER_BLOCK < batch_size; ++i) {
			int64_t batch_index = i + blockIdx.x*BATCH_PER_BLOCK;
			T2 my_a = ((const T2*)a)[batch_index*hidden_size/2 + threadIdx.x];
			T2 my_b = ((const T2*)b)[batch_index*hidden_size/2 + threadIdx.x];
			T2 my_output = __hfma2(__hadd2(my_a, my_b), my_c, my_d);
			((T2*)output)[batch_index*hidden_size/2 + threadIdx.x] = my_output;
		}
	}
}

// fused_batched_add_mult_add: Calculate (a + b) * c + d
// A and b are batched (of shape [batch_size, hidden_size])
// c and d are not batched (of shape [hidden_size])
torch::Tensor fused_batched_add_mult_add(
	torch::Tensor a,	// [batch_size, hidden_size]
	torch::Tensor b,	// [batch_size, hidden_size]
	torch::Tensor c,	// [hidden_size]
	torch::Tensor d		// [hidden_size]
) {
	assert (c.is_contiguous());
	assert (d.is_contiguous());
	if (!a.is_contiguous()) {
		a = a.contiguous();
	}
	if (!b.is_contiguous()) {
		b = b.contiguous();
	}

	if (a.dtype() != b.dtype() || a.dtype() != c.dtype() || a.dtype() != d.dtype()) {
		throw std::runtime_error("All inputs must have the same dtype");
	}

	torch::Tensor output = torch::empty_like(a).contiguous();
	int64_t hidden_size = a.size(-1);
	int64_t batch_size = a.numel() / hidden_size;

	if (hidden_size % 2 != 0) {
		throw std::runtime_error("hidden_size must be divisible by 2");
	}

	int64_t grid_size = (batch_size+BATCH_PER_BLOCK-1)/BATCH_PER_BLOCK;
	int64_t block_size = hidden_size/2;

	if (a.dtype() == torch::kFloat16) {
		fused_add_mult_add_kernel<half><<<grid_size, block_size>>>(
			(half*)output.data_ptr(),
			(const half*)a.data_ptr(),
			(const half*)b.data_ptr(),
			(const half*)c.data_ptr(),
			(const half*)d.data_ptr(),
			batch_size,
			hidden_size
		);
	} else {
		throw std::runtime_error("Only float16 is supported");
	}

	return output;
}
