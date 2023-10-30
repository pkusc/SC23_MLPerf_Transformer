#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_fp16.h>

#include "mobilebert_cpp_kernels.h"

namespace py = pybind11;

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

	if (a.dtype() == torch::kFloat16) {
		invoke_fused_add_mult_add(
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


torch::Tensor batched_masked_self_attention(
	torch::Tensor query,	// [batch_size, context_len, num_heads, head_size]
	torch::Tensor key,		// [batch_size, context_len, num_heads, head_size]
	torch::Tensor value,	// [batch_size, context_len, num_heads, head_size]
	torch::Tensor attention_mask_prefix_lens	// [batch_size]
	// Return: [batch_size, context_len, num_heads, head_size]
) {
	int64_t batch_size = query.size(0);
	int64_t context_len = query.size(1);
	int64_t num_heads = query.size(2);
	int64_t head_size = query.size(3);
	// query = query.permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, context_len, head_size});
	// key = key.permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, context_len, head_size});
	// value = value.permute({0, 2, 1, 3}).contiguous().view({batch_size * num_heads, context_len, head_size});
	// // qkv: [batch_size * num_heads, context_len, head_size]
	// torch::Tensor attention_scores = torch::matmul(query, key.transpose(1, 2));
	// attention_scores /= std::sqrt((float)head_size);
	// // attention_scores: [batch_size * num_heads, context_len, context_len]
	// torch::Tensor attention_masks = torch::zeros({batch_size * num_heads, context_len, context_len}, query.device()).to(query.dtype());
	// // attention_masks: [batch_size * num_heads, context_len, context_len]
	// for (int64_t i = 0; i < batch_size; i++) {
	// 	int64_t prefix_len = attention_mask_prefix_lens[i].item<int64_t>();
	// 	for (int j = 0; j < num_heads; j++) {
	// 		attention_masks[i * num_heads + j].slice(1, prefix_len, 10000000).fill_(-65500.0);
	// 	}
	// }
	// attention_scores += attention_masks;
	// attention_scores = torch::softmax(attention_scores, 2);
	// // attention_scores: [batch_size * num_heads, context_len, context_len]
	// torch::Tensor attention_output = torch::matmul(attention_scores, value);
	// // attention_output: [batch_size * num_heads, context_len, head_size]
	// attention_output = attention_output.view({batch_size, num_heads, context_len, head_size}).permute({0, 2, 1, 3}).contiguous();
	// // attention_output: [batch_size, context_len, num_heads, head_size]
	if (!query.is_contiguous()) {
		query = query.contiguous();
	}
	if (!key.is_contiguous()) {
		key = key.contiguous();
	}
	if (!value.is_contiguous()) {
		value = value.contiguous();
	}
	if (!attention_mask_prefix_lens.is_contiguous()) {
		attention_mask_prefix_lens = attention_mask_prefix_lens.contiguous();
	}
	torch::Tensor attention_output = torch::zeros_like(query);
	assert (attention_output.is_contiguous());
	assert (query.dtype() == torch::kFloat16);
	assert (key.dtype() == torch::kFloat16);
	assert (value.dtype() == torch::kFloat16);
	assert (attention_mask_prefix_lens.dtype() == torch::kInt32);

	invoke_batched_masked_self_attention(
		(half*)attention_output.data_ptr(),
		(const half*)query.data_ptr(),
		(const half*)key.data_ptr(),
		(const half*)value.data_ptr(),
		(const int*)attention_mask_prefix_lens.data_ptr(),
		batch_size,
		context_len,
		num_heads,
		head_size
	);

	return attention_output;
}

class FusedAddResidualNoNorm : public torch::CustomClassHolder {
public:
	torch::Tensor weight;
	torch::Tensor bias;

	FusedAddResidualNoNorm(
		torch::Tensor weight,
		torch::Tensor bias) :
		weight(weight.contiguous()),
		bias(bias.contiguous()){}
	
	torch::Tensor forward(torch::Tensor input, torch::Tensor residual) {
		// return (input + residual) * weight + bias;
		return fused_batched_add_mult_add(input, residual, weight, bias);
	}
};

PYBIND11_MODULE(mobilebert_cpp, m) {
	py::class_<FusedAddResidualNoNorm>(m, "FusedAddResidualNoNorm")
		.def(py::init<torch::Tensor, torch::Tensor>())
		.def("forward", &FusedAddResidualNoNorm::forward);
	m.def("batched_masked_self_attention", &batched_masked_self_attention);
}