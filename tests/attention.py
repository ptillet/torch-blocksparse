import torch_blocksparse
import torch

use_half = False
BatchSize, NumHeads, SeqLen, Embed = 32, 16, 128, 1024
#BatchSize, NumHeads, SeqLen, Embed = 8, 32, 512, 1024
#BatchSize, NumHeads, SeqLen, Embed = 16, 16, 1024, 1024
#BatchSize, NumHeads, SeqLen, Embed = 8, 16, 4096, 1024

# create sparse multi-head attention module
sparsity = torch_blocksparse.MultiheadAttention.SparsityInfo()
sparsity.mode = 'dense'
sparsity.block = 16
torch.manual_seed(0)
sparse_mha = torch_blocksparse.MultiheadAttention(Embed, NumHeads, sparsity, key_padding_mask_mode='add', attn_mask_mode='mul').cuda()
# create dense multi-head attention module
torch.manual_seed(0)
dense_mha  = torch.nn.modules.MultiheadAttention(Embed, NumHeads).cuda()
# test
query      = torch.rand(SeqLen, BatchSize, Embed).cuda()
key        = torch.rand(SeqLen, BatchSize, Embed).cuda()
value      = torch.rand(SeqLen, BatchSize, Embed).cuda()
# key-padding mask
torch_kp_mask = torch.randint(0, 2, (BatchSize, SeqLen), dtype=torch.bool).cuda()
triton_kp_mask = torch_kp_mask.type(torch.float32)
triton_kp_mask[triton_kp_mask==1.] = float('-inf')
# attention mask
triton_attn_mask = torch.randint(0, 2, (SeqLen, SeqLen), dtype=torch.float32).cuda()
torch_attn_mask = 1 - triton_attn_mask
torch_attn_mask[torch_attn_mask==1] = float('-inf')
# to half precision
if use_half:
    sparse_mha = sparse_mha.half()
    dense_mha = dense_mha.half()
    query = query.half()
    key = key.half()
    value = value.half()
    triton_kp_mask = triton_kp_mask.half()
# run modules
sparse_out, _ = sparse_mha(query, key, value, key_padding_mask=triton_kp_mask, attn_mask=triton_attn_mask, need_weights=False)
dense_out, _ = dense_mha(query, key, value, key_padding_mask=torch_kp_mask, attn_mask=torch_attn_mask, need_weights=False)
if use_half:
    assert torch.allclose(sparse_out, dense_out, rtol=1e-3, atol=1e-3)
else:
    assert torch.allclose(sparse_out, dense_out, rtol=1e-5, atol=1e-6)