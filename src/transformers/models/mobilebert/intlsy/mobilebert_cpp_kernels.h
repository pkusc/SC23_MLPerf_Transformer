#pragma once

#include <torch/extension.h>

// fused_batched_add_mult_add: Calculate (a + b) * c + d
torch::Tensor fused_batched_add_mult_add(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor c,
	torch::Tensor d
);
