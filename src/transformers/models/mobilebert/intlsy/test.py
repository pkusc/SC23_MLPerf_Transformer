import os, sys, math

import torch
import mobilebert_cpp

def reference_impl(
	query: torch.Tensor,
	key: torch.Tensor,
	value: torch.Tensor,
	attention_mask_prefix_lens: torch.Tensor,
) -> torch.Tensor:
	batch_size = query.shape[0]
	context_len = query.shape[1]
	num_heads = query.shape[2]
	head_size = query.shape[3]

	query = query.permute(0, 2, 1, 3).contiguous().view(batch_size * num_heads, context_len, head_size)
	key = key.permute(0, 2, 1, 3).contiguous().view(batch_size * num_heads, context_len, head_size)
	value = value.permute(0, 2, 1, 3).contiguous().view(batch_size * num_heads, context_len, head_size)

	attention_scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(head_size))
	attention_scores = attention_scores.view(batch_size, num_heads, context_len, context_len)

	attention_masks = torch.zeros_like(attention_scores)
	for i in range(batch_size):
		prefix_len = attention_mask_prefix_lens[i]
		attention_masks[i, :, :, prefix_len:] = -65504
	attention_scores += attention_masks
	# return attention_scores.view(batch_size, num_heads, context_len, context_len).permute(0, 2, 1, 3).contiguous().view(batch_size, context_len, num_heads, context_len)
	attention_scores = torch.softmax(attention_scores, dim=-1)
	attention_output = torch.bmm(attention_scores.view(batch_size * num_heads, context_len, context_len), value.view(batch_size * num_heads, context_len, head_size))
	attention_output = attention_output.view(batch_size, num_heads, context_len, head_size).permute(0, 2, 1, 3).contiguous().view(batch_size, context_len, num_heads, head_size)
	return attention_output

batch_size = 1
context_len = 32
num_heads = 4
head_size = 32

torch.set_default_device('cuda')
torch.set_default_dtype(torch.float16)
torch.manual_seed(0)

query = torch.rand((batch_size, context_len, num_heads, head_size))
key = torch.rand((batch_size, context_len, num_heads, head_size))
value = torch.rand((batch_size, context_len, num_heads, head_size))
attention_mask_prefix_lens = torch.randint(max(context_len-100, 10), context_len, (batch_size,)).to(torch.int32)

reference_ans = reference_impl(query, key, value, attention_mask_prefix_lens)
my_ans = mobilebert_cpp.batched_masked_self_attention(query, key, value, attention_mask_prefix_lens)
torch.cuda.synchronize()

torch.set_default_dtype(torch.float32)
reference_ans_fp32 = reference_impl(
	query.to(torch.float32),
	key.to(torch.float32),
	value.to(torch.float32),
	attention_mask_prefix_lens
)

# torch.set_printoptions(threshold=20000, linewidth=1000)
# print(reference_ans)
# print(my_ans)
# print(reference_ans - my_ans)

max_abs_err = torch.max(torch.abs(reference_ans_fp32 - reference_ans))
print("Max abs err (std vs std+fp32):", max_abs_err)
max_rel_err = torch.max(torch.abs(reference_ans_fp32 - reference_ans) / torch.abs(reference_ans_fp32))
print("Max rel err (std vs std+fp32):", max_rel_err)

max_abs_err = torch.max(torch.abs(reference_ans - my_ans))
print("Max abs err:", max_abs_err)
max_rel_err = torch.max(torch.abs(reference_ans - my_ans) / torch.abs(reference_ans))
print("Max rel err:", max_rel_err)

max_abs_err = torch.max(torch.abs(reference_ans_fp32 - my_ans))
print("Max abs err (with std+fp32):", max_abs_err)
max_rel_err = torch.max(torch.abs(reference_ans_fp32 - my_ans) / torch.abs(reference_ans_fp32))
print("Max rel err (with std+fp32):", max_rel_err)

if max_abs_err > 0.01 or max_rel_err > 0.01 or math.isnan(max_abs_err.item()) or math.isnan(max_rel_err.item()):
	print("ERROR: max_abs_err or max_rel_err is not legal!")
	sys.exit(1)

abs_err_map = (reference_ans - my_ans)[0, :128, 0, :]
rel_err_map = (reference_ans - my_ans)[0, :128, 0, :] / reference_ans[0, :128, 0, :]

# Draw a heat map of abs_err_map and rel_err_map
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(abs_err_map.cpu().numpy(), cmap='coolwarm', interpolation='nearest')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(rel_err_map.cpu().numpy(), cmap='coolwarm', interpolation='nearest')
# plt.colorbar()
# plt.show()
