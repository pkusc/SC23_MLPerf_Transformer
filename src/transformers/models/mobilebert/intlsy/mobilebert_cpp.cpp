#include <torch/extension.h>

#include <iostream>
#include <vector>

#include "mobilebert_cpp_kernels.h"

namespace py = pybind11;

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
}