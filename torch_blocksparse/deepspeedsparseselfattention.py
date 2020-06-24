import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import torch_blocksparse
import sys

class SparsityConfig:
    def __init__(self,
            mode = 'fixed',
            block = 16,
            stride = 64,
            attention = 'unidirectional',
            numverts = 1,
            vertsize = 1):
        if mode != 'dense' and mode != 'fixed':
            raise NotImplementedError('only \"dense\" and \"fixed\" modes are supported for now')
        self.mode = mode
        self.block = block
        self.stride = stride
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only \"uni/bi-directional\" attentions are supported for now')
        self.attention = attention
        self.numverts = numverts
        self.vertsize = vertsize


class DeepSpeedSparseSelfAttention(nn.Module):

    # Make binary block-sparsity layout from given parameters
    # contribution of Arash Ashari (Microsoft Research)
    @staticmethod
    def _set_s1_layout(layout, h, num_blocks, block_stride, attention):
        for i in range(0, num_blocks, block_stride):
            for j in range(i, i + block_stride):
                for k in range(i, (j + 1 if attention == 'unidirectional' else i + block_stride)):
                    layout[h, j, k] = 1
        return layout

    @staticmethod
    def _set_s2_layout(layout, h, num_blocks, block_stride, attention, numverts, vertsize):
        start = block_stride - (1 + h % numverts) * vertsize
        for i in range(0, num_blocks):
            end = i if attention == 'unidirectional' else num_blocks
            for j in range(start, end, block_stride):
                for k in range(j, min(j + vertsize, num_blocks)):
                    layout[h, i, k] = 1
        return layout

    @staticmethod
    def _make_layout(num_heads, num_blocks, mode, block_stride, attention, numverts, vertsize):
        if (block_stride / vertsize) != (block_stride // vertsize):
                raise ValueError(f'Number of blocks in a stride window {block_stride} must be dividable by vertical block size {vertsize}')
        
        if numverts > (block_stride / vertsize):
                raise ValueError(f'Number of layout versions {num_verts} cannot be larger than blocks in a stride window divided by vertical block size {block_stride} / {vertsize} = {block_stride/vertsize}')

        layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        if mode == "dense":
            layout[:, :, :] = 1
        elif mode == "fixed":
            for i in range(0, num_heads):
                layout = DeepSpeedSparseSelfAttention._set_s1_layout(layout, i, num_blocks, block_stride, attention)
                layout = DeepSpeedSparseSelfAttention._set_s2_layout(layout, i, num_blocks, block_stride, attention, numverts, vertsize)
        return layout

    ops = dict()

    # add to cache
    def get_ops(self, H, L):
        import sys
        if L not in DeepSpeedSparseSelfAttention.ops:
            spConfig = self.sparsity_config

            num_blocks = L // spConfig.block
            if num_blocks != L / spConfig.block:
                raise ValueError(f'Sequence length {L} must be dividable by block size {spConfig.block}')

            block_stride = spConfig.stride // spConfig.block
            if block_stride != spConfig.stride // spConfig.block:
                raise ValueError(f'Stride {spConfig.stride} must be dividable by block size {spConfig.block}')

            layout = DeepSpeedSparseSelfAttention._make_layout(H,
                    num_blocks,
                    spConfig.mode,
                    block_stride,
                    spConfig.attention,
                    spConfig.numverts, 
                    spConfig.vertsize)

            sparse_dot_sdd_nt = torch_blocksparse.MatMul(layout,
                    spConfig.block,
                    'sdd',
                    trans_a=False,
                    trans_b=True)

            sparse_dot_dsd_nn = torch_blocksparse.MatMul(layout,
                    spConfig.block,
                    'dsd',
                    trans_a=False,
                    trans_b=False)

            sparse_softmax = torch_blocksparse.Softmax(layout, spConfig.block)

            DeepSpeedSparseSelfAttention.ops[L] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax)
        return DeepSpeedSparseSelfAttention.ops[L]

    # constructor
    def __init__(self, sparsity_config=SparsityConfig(), key_padding_mask_mode='add', attn_mask_mode='mul'):
        super().__init__()

        # sparsity information
        self.sparsity_config = sparsity_config
       
        # mask modes
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self, query, key, value, rpe=None, key_padding_mask=None, attn_mask=None):
        bsz, num_heads, tgt_len, head_dim = query.size()
        
        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)

        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype, key_padding_mask, is_key_padding_mask=True)


        # squeeze attn_mask if it is given
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)

        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)

        scaling = float(head_dim) ** -0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling, rpe=rpe, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)

        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output
