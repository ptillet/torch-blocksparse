import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import torch_blocksparse
import sys


class SimplifiedSparseAttention(nn.Module):

    # Make binary block-sparsity layout from given parameters
    # contribution of Arash Ashari (Microsoft Research)
    @staticmethod
    def _set_s1_layout(layout, h, num_blocks, block_stride, unidirectional):
        for i in range(0, num_blocks, block_stride):
            for j in range(i, i + block_stride):
                for k in range(i, (j + 1 if unidirectional else i + block_stride)):
                    layout[h, j, k] = 1
        return layout

    @staticmethod
    def _set_s2_layout(layout, h, num_blocks, block_stride, unidirectional, numverts, vertsize):
        start = block_stride - (1 + h % numverts) * vertsize
        for i in range(0, num_blocks):
            end = i if unidirectional else num_blocks
            for j in range(start, end, block_stride):
                for k in range(j, min(j + vertsize, num_blocks)):
                    layout[h, i, k] = 1
        return layout

    @staticmethod
    def _make_layout(num_heads, num_blocks, mode, block_stride, unidirectional, numverts, vertsize):
        layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        if mode == "dense":
            layout[:, :, :] = 1
        elif mode == "fixed":
            for i in range(0, num_heads):
                layout = SimplifiedSparseAttention._set_s1_layout(layout, i, num_blocks, block_stride, unidirectional)
                layout = SimplifiedSparseAttention._set_s2_layout(layout, i, num_blocks, block_stride, unidirectional, numverts, vertsize)
        return layout

    class SparsityInfo:

        def __init__(self, mode = None,
                     block = None, stride=128,
                     unidirectional = None, numverts = 1, vertsize = 1):
            self.mode = mode
            self.block = block
            self.stride = stride
            self.unidirectional = unidirectional
            self.numverts = numverts
            self.vertsize = vertsize

    ops = dict()

    # add to cache
    def get_ops(self, L):
        import sys
        if L not in SimplifiedSparseAttention.ops:
            sparsity = self.sparsity
            layout = SimplifiedSparseAttention._make_layout(self.num_heads, L // sparsity.block, sparsity.mode,
                                                    sparsity.stride // sparsity.block, sparsity.unidirectional,
                                                    sparsity.numverts, sparsity.vertsize)
            sparse_dot_sdd_nt = torch_blocksparse.MatMul(layout, sparsity.block, 'sdd',
                                                               trans_a=False, trans_b=True)
            sparse_dot_dsd_nn = torch_blocksparse.MatMul(layout, sparsity.block, 'dsd',
                                                               trans_a=False, trans_b=False)
            sparse_softmax = torch_blocksparse.Softmax(layout, sparsity.block)
            SimplifiedSparseAttention.ops[L] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax)
        return SimplifiedSparseAttention.ops[L]

    # constructor
    def __init__(self, embed_dim, num_heads, sparsity, key_padding_mask_mode='add', attn_mask_mode='mul'):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparsity = sparsity
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode


    # forward pass
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')
        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(query.shape[2])

        bsz, num_heads, tgt_len, head_dim = query.size()
        scaling = float(head_dim) ** -0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)
        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output
