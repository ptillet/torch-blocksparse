import torch_blocksparse
import torch

L, S, N, E = 128, 128, 2, 512
num_heads = 2
# create sparse multi-head attention module
sparsity = torch_blocksparse.MultiheadAttention.SparsityInfo()
sparsity.mode = 'dense'
sparsity.block = 16
torch.manual_seed(0)
sparse_mha = torch_blocksparse.MultiheadAttention(512, num_heads, L, sparsity).cuda()
# create dense multi-head attention module
torch.manual_seed(0)
dense_mha  = torch.nn.modules.MultiheadAttention(512, num_heads) .cuda()
# test
query      = torch.rand(L, N, E).cuda()
key        = torch.rand(S, N, E).cuda()
value      = torch.rand(S, N, E).cuda()
mul_mask   = torch.randint(0, 2, (N, S), dtype=torch.bool).cuda()
add_mask   = mul_mask.type(torch.float32)
add_mask[add_mask==1.] = float('-inf')

sparse_out, _ = sparse_mha(query, key, value, key_padding_mask=add_mask, need_weights=False)
dense_out, _ = dense_mha(query, key, value, key_padding_mask=mul_mask, need_weights=False)
print((sparse_out - dense_out).abs().max())
assert torch.allclose(sparse_out, dense_out, atol=1e-6)