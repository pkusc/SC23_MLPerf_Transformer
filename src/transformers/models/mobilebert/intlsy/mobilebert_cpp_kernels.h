#pragma once

// fused_batched_add_mult_add: Calculate (a + b) * c + d
template<typename T>
void invoke_fused_add_mult_add(
	T* output,
	const T* a,
	const T* b,
	const T* c,
	const T* d,
	int64_t batch_size,
	int64_t hidden_size
);

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
);
