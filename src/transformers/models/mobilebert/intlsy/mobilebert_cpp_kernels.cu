#include "mobilebert_cpp_kernels.h"

#include <stdexcept>
#include <cstdio>
#include <cassert>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>	// For Tensor Core
using namespace nvcuda;

static constexpr int64_t BATCH_PER_BLOCK = 16;

// Block shape: [hidden_size/2]
template<typename T>
__global__ void fused_add_mult_add_kernel(
	T* __restrict__ output,		// [batch_size, hidden_size]
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

template<typename T>
void invoke_fused_add_mult_add(
	T* output,
	const T* a,
	const T* b,
	const T* c,
	const T* d,
	int64_t batch_size,
	int64_t hidden_size
) {
	int64_t grid_size = (batch_size+BATCH_PER_BLOCK-1)/BATCH_PER_BLOCK;
	int64_t block_size = hidden_size/2;

	fused_add_mult_add_kernel<T><<<grid_size, block_size>>>(
		output,
		a,
		b,
		c,
		d,
		batch_size,
		hidden_size
	);
}

template void invoke_fused_add_mult_add<half>(
	half* output,
	const half* a,
	const half* b,
	const half* c,
	const half* d,
	int64_t batch_size,
	int64_t hidden_size
);


template<typename T>
__device__ __forceinline__ T my_max(const T &a, const T &b) {
	return a > b ? a : b;
}

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) \
    (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

// Grid shape: [batch_size, num_heads]
// Block shape: [CONTEXT_LEN/16*32]
template<
	typename T,
	int64_t CONTEXT_LEN,
	int64_t NUM_HEADS,
	int64_t HEAD_SIZE
>
__global__ void batched_masked_self_attention_kernel(
	T* __restrict__ result,	// [batch_size, context_len, num_heads, head_size]
	const T* __restrict__ query,	// [batch_size, context_len, num_heads, head_size]
	const T* __restrict__ key,		// [batch_size, context_len, num_heads, head_size]
	const T* __restrict__ value,	// [batch_size, context_len, num_heads, head_size]
	const int32_t* __restrict__ attention_mask_prefix_lens	// [batch_size]
) {
	static constexpr int64_t TILE_SIZE = 16;	// WMMA block size
	static constexpr int64_t WARP_SIZE = 32;
	static constexpr int64_t NUM_WARPS = CONTEXT_LEN / TILE_SIZE;
	const int64_t warp_id = threadIdx.x / WARP_SIZE;
	const int64_t lane_id = threadIdx.x % WARP_SIZE;
	const int64_t batch_id = blockIdx.x;
	const int64_t head_id = blockIdx.y;
	const float qk_scale = 1.0f / sqrtf((float)HEAD_SIZE);

	// To process every elements (16*16 = 256 elements in total), we 
	// group threads in the same warp into thread groups. Every thread
	// group contains two threads. Thread 0 and 16, thread 1 and 17, and
	// so on, are in the same group. Each thread group is assigned with
	// one row to proceed.
	static constexpr int64_t THREAD_GROUP_SIZE = 2;
	const int64_t thread_group_id = lane_id & 0x0Fu;	// The row, i.e. the thread group id
	const int64_t thread_group_offset = lane_id >> 4;	// The offset in the thread group (0 or 1)

	static_assert(std::is_same<T, __half>::value, "Only half is supported");
	static_assert(HEAD_SIZE == 32, "Only head size 32 is supported");
	static_assert(CONTEXT_LEN % TILE_SIZE == 0, "Context length must be divisible by 16");

	// Fragments to store qkv tiles
	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, wmma::row_major> q_frag_1;
	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, wmma::row_major> q_frag_2;
	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, wmma::col_major> k_frag;
	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, wmma::row_major> v_frag;
	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> qk_acc_frag;
	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, T, wmma::col_major> qk_data_frag;
	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, T> kv_acc_frag;

	__shared__ T k_tile[TILE_SIZE*HEAD_SIZE], v_tile[TILE_SIZE*HEAD_SIZE];
	__shared__ T acc_tile[NUM_WARPS*TILE_SIZE*HEAD_SIZE];

	__shared__ float m_buf[CONTEXT_LEN], l_buf[CONTEXT_LEN];
	__shared__ T mult_old_elem[CONTEXT_LEN], mult_new_elem[CONTEXT_LEN];

	T* my_temp_result_tile = (T*)(acc_tile + warp_id * TILE_SIZE * HEAD_SIZE);						// [TILE_SIZE, HEAD_SIZE]
	float* my_acc_tile = (float*)my_temp_result_tile;					// [TILE_SIZE, TILE_SIZE]
	float* my_m_buf = m_buf + warp_id * TILE_SIZE;						// [TILE_SIZE]
	float* my_l_buf = l_buf + warp_id * TILE_SIZE;						// [TILE_SIZE]
	T* my_mult_old_elem = mult_old_elem + warp_id * TILE_SIZE;			// [TILE_SIZE]
	T* my_mult_new_elem = mult_new_elem + warp_id * TILE_SIZE;			// [TILE_SIZE]
	const int64_t my_attention_mask_prefix_lens = attention_mask_prefix_lens[batch_id];

	if (lane_id < TILE_SIZE) {
		my_m_buf[lane_id] = -60000.0f;
		my_l_buf[lane_id] = 0.0f;
	}

	wmma::load_matrix_sync(
		q_frag_1,
		query + INDEX_4D(0, CONTEXT_LEN, NUM_HEADS, HEAD_SIZE,
						 batch_id, warp_id * TILE_SIZE, head_id, 0),
		HEAD_SIZE*NUM_HEADS
	);

	wmma::load_matrix_sync(
		q_frag_2,
		query + INDEX_4D(0, CONTEXT_LEN, NUM_HEADS, HEAD_SIZE,
						 batch_id, warp_id * TILE_SIZE, head_id, TILE_SIZE),
		HEAD_SIZE*NUM_HEADS
	);

	for (int64_t tile_start = 0; tile_start < my_attention_mask_prefix_lens; tile_start += TILE_SIZE) {
		// if XXXX break

		// Step 1. Load k_frag and v_frag
		__syncthreads();
		if (warp_id == 0) {
			#pragma unroll
			for (int64_t i = 0; i < TILE_SIZE; i += 1) {
				k_tile[i*HEAD_SIZE+lane_id] = key[INDEX_4D(0, CONTEXT_LEN, NUM_HEADS, HEAD_SIZE,
												batch_id, tile_start+i, head_id, lane_id)];
				v_tile[i*HEAD_SIZE+lane_id] = value[INDEX_4D(0, CONTEXT_LEN, NUM_HEADS, HEAD_SIZE,
												batch_id, tile_start+i, head_id, lane_id)];
			}
		}
		__syncthreads();

		wmma::fill_fragment(qk_acc_frag, (T)0.0);
		wmma::load_matrix_sync(
			k_frag,
			k_tile,
			HEAD_SIZE
		);
		wmma::mma_sync(qk_acc_frag, q_frag_1, k_frag, qk_acc_frag);

		wmma::load_matrix_sync(
			k_frag,
			k_tile + TILE_SIZE,
			HEAD_SIZE
		);
		wmma::mma_sync(qk_acc_frag, q_frag_2, k_frag, qk_acc_frag);

		// Store Q_i*K_j^T*qk_scale to cur_qk_tile
		// Since the following ops are mainly row-wise ops, to avoid
		// reduction between threads and shared memory bank conflicts,
		// we store cur_qk_frag in a transposed way.
		wmma::store_matrix_sync(
			my_acc_tile,
			qk_acc_frag,
			TILE_SIZE,
			wmma::mem_col_major
		);

		// Scale & Mask
		#pragma unroll
		for (int64_t col = thread_group_offset; col < TILE_SIZE; col += THREAD_GROUP_SIZE) {
			float x = my_acc_tile[INDEX_2D(TILE_SIZE, TILE_SIZE, col, thread_group_id)];
			x *= qk_scale;
			x += tile_start+col >= my_attention_mask_prefix_lens ? -65504.0f : 0.0f;
			my_acc_tile[INDEX_2D(TILE_SIZE, TILE_SIZE, col, thread_group_id)] = x;
		}


		// Step 7. Compute:
		// - mij (containing the maximum value in every row)
		// - lij (containing the sum of exp(Sij-mij) in every row)
		{
			float mij = -60000.;
			#pragma unroll
			for (int64_t col = thread_group_offset; col < TILE_SIZE; col += THREAD_GROUP_SIZE)
				mij = my_max(mij, my_acc_tile[INDEX_2D(TILE_SIZE, TILE_SIZE, col, thread_group_id)]);
			mij = my_max(mij, __shfl_xor_sync(0xffffffff, mij, 16));
			float lij = (T)0.0;
			#pragma unroll
			for (int64_t col = thread_group_offset; col < TILE_SIZE; col += THREAD_GROUP_SIZE) {
				float x = __expf((float)(my_acc_tile[INDEX_2D(TILE_SIZE, TILE_SIZE, col, thread_group_id)] - mij));
				lij += x;
				my_temp_result_tile[INDEX_2D(TILE_SIZE, HEAD_SIZE, col, thread_group_id)] = x;
			}
			lij += __shfl_xor_sync(0xffffffff, lij, 16);

			float mi = my_m_buf[thread_group_id];
			float li = my_l_buf[thread_group_id];
			float mi_new = my_max(mi, mij);
			float li_new = __expf(mi-mi_new)*li + __expf(mij-mi_new)*lij;
			my_mult_old_elem[thread_group_id] = (T)(li * (__expf(mi-mi_new) / li_new));
			my_mult_new_elem[thread_group_id] = (T)(__expf(mij-mi_new) / li_new);

			my_m_buf[thread_group_id] = mi_new;
			my_l_buf[thread_group_id] = li_new;
		}

		wmma::load_matrix_sync(
			qk_data_frag,
			my_temp_result_tile,
			HEAD_SIZE
		);

		wmma::fill_fragment(kv_acc_frag, (T)0.0);
		wmma::load_matrix_sync(
			v_frag,
			v_tile,
			HEAD_SIZE
		);
		wmma::mma_sync(kv_acc_frag, qk_data_frag, v_frag, kv_acc_frag);
		wmma::store_matrix_sync(
			my_temp_result_tile,
			kv_acc_frag,
			HEAD_SIZE,
			wmma::mem_row_major
		);

		wmma::fill_fragment(kv_acc_frag, (T)0.0);
		wmma::load_matrix_sync(
			v_frag,
			v_tile + TILE_SIZE,
			HEAD_SIZE
		);
		wmma::mma_sync(kv_acc_frag, qk_data_frag, v_frag, kv_acc_frag);
		wmma::store_matrix_sync(
			my_temp_result_tile+TILE_SIZE,
			kv_acc_frag,
			HEAD_SIZE,
			wmma::mem_row_major
		);

		#pragma unroll
		for (int64_t row = thread_group_offset; row < TILE_SIZE; row += THREAD_GROUP_SIZE) {
			int64_t result_index = INDEX_4D(
				0, CONTEXT_LEN, NUM_HEADS, HEAD_SIZE/2,
				batch_id, warp_id*TILE_SIZE+row, head_id, thread_group_id
			);
			half2 cur_result = ((half2*)result)[result_index];
			half2 new_result = ((half2*)my_temp_result_tile)[INDEX_2D(TILE_SIZE, HEAD_SIZE/2, row, thread_group_id)];
			((half2*)result)[result_index] = (half2){
				my_mult_old_elem[row]*cur_result.x + my_mult_new_elem[row]*new_result.x,
				my_mult_old_elem[row]*cur_result.y + my_mult_new_elem[row]*new_result.y
			};
		}
	}
}

template<typename T>
void invoke_batched_masked_self_attention(
	T* __restrict__ result,	// [batch_size, context_len, num_heads, head_size]
	const T* __restrict__ query,	// [batch_size, context_len, num_heads, head_size]
	const T* __restrict__ key,		// [batch_size, context_len, num_heads, head_size]
	const T* __restrict__ value,	// [batch_size, context_len, num_heads, head_size]
	const int32_t* __restrict__ attention_mask_prefix_lens,	// [batch_size]
	int64_t batch_size,
	int64_t context_len,
	int64_t num_heads,
	int64_t head_size
) {
	assert (context_len == 32 || context_len == 384);
	assert (head_size == 32);
	assert (num_heads == 4);
	dim3 grid_dim = dim3(batch_size, num_heads);
	dim3 block_dim = dim3(context_len / 16 * 32);
	if (context_len == 32) {
		batched_masked_self_attention_kernel<T, 32, 4, 32><<<grid_dim, block_dim>>>(
			result,
			query,
			key,
			value,
			attention_mask_prefix_lens
		);
	} else {
		batched_masked_self_attention_kernel<half, 384, 4, 32><<<grid_dim, block_dim>>>(
			result,
			query,
			key,
			value,
			attention_mask_prefix_lens
		);
	}
}

template void invoke_batched_masked_self_attention<half>(
	half* __restrict__ result,	// [batch_size, context_len, num_heads, head_size]
	const half* __restrict__ query,	// [batch_size, context_len, num_heads, head_size]
	const half* __restrict__ key,		// [batch_size, context_len, num_heads, head_size]
	const half* __restrict__ value,	// [batch_size, context_len, num_heads, head_size]
	const int32_t* __restrict__ attention_mask_prefix_lens,	// [batch_size]
	int64_t batch_size,
	int64_t context_len,
	int64_t num_heads,
	int64_t head_size
);
