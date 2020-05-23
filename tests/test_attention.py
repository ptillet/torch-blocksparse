import torch_blocksparse
import torch
import multiprocessing
from nose.tools import nottest
from parameterized import parameterized


def task(query, key, value, add_mask, Embed, NumHeads, sparsity):
    mha = torch_blocksparse.MultiheadAttention(Embed, NumHeads, sparsity).cuda()
    mha(query, key, value, key_padding_mask=add_mask, need_weights=False)

def test_main():
    torch.multiprocessing.freeze_support()

    use_half = False
    BatchSize, NumHeads, SeqLen, Embed = 2, 2, 128, 128
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
    dense_mha  = torch.nn.modules.MultiheadAttention(Embed, NumHeads).cuda()
    # test
    query      = torch.rand(SeqLen, BatchSize, Embed).cuda()
    key        = torch.rand(SeqLen, BatchSize, Embed).cuda()
    value      = torch.rand(SeqLen, BatchSize, Embed).cuda()
    mul_mask   = torch.randint(0, 2, (BatchSize, SeqLen), dtype=torch.bool).cuda()
    add_mask   = mul_mask.type(torch.float32)
    add_mask[add_mask==1.] = float('-inf')
    # to half precision
    if use_half:
        sparse_mha = sparse_mha.half()
        dense_mha = dense_mha.half()
        query = query.half()
        key = key.half()
        value = value.half()
        add_mask = add_mask.half()
    # run modules
    #ctx = multiprocessing.get_context('spawn')
    #procs = [ctx.Process(target=task, args=(query,key,value,add_mask, Embed, NumHeads, sparsity)) for i in range(2)]
    #for p in procs:
    #    p.start()
    #for p in procs:
    #    p.join()
    sparse_out, _ = sparse_mha(query, key, value, key_padding_mask=add_mask, need_weights=False)
    dense_out, _ = dense_mha(query, key, value, key_padding_mask=mul_mask, need_weights=False)
    if use_half:
        assert torch.allclose(sparse_out, dense_out, rtol=1e-3, atol=1e-3)
    else:
        assert torch.allclose(sparse_out, dense_out, rtol=1e-5, atol=1e-6)