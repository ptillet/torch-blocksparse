import torch_blocksparse
import torch

#BatchSize, NumHeads, SeqLen, Embed = 32, 16, 128, 1024
#BatchSize, NumHeads, SeqLen, Embed = 8, 32, 512, 1024
#BatchSize, NumHeads, SeqLen, Embed = 16, 16, 1024, 1024
#BatchSize, NumHeads, SeqLen, Embed = 8, 16, 4096, 1024

# create sparse multi-head attention module
sparsity = torch_blocksparse.MultiheadAttention.SparsityInfo()
sparsity.mode = 'dense'
sparsity.block = 16
torch.manual_seed(0)
sparse_mha = torch_blocksparse.MultiheadAttention(Embed, NumHeads, sparsity).cuda()
# create dense multi-head attention module
torch.manual_seed(0)
dense_mha  = torch.nn.modules.MultiheadAttention(Embed, NumHeads) .cuda()
# test
query      = torch.rand(SeqLen, BatchSize, Embed).cuda()
key        = torch.rand(SeqLen, BatchSize, Embed).cuda()
value      = torch.rand(SeqLen, BatchSize, Embed).cuda()
mul_mask   = torch.randint(0, 2, (BatchSize, SeqLen), dtype=torch.bool).cuda()
add_mask   = mul_mask.type(torch.float32)
add_mask[add_mask==1.] = float('-inf')

sparse_out, _ = sparse_mha(query, key, value, key_padding_mask=add_mask, need_weights=False)
dense_out, _ = dense_mha(query, key, value, key_padding_mask=mul_mask, need_weights=False)
print((sparse_out - dense_out).abs().max())
assert torch.allclose(sparse_out, dense_out, atol=1e-6)