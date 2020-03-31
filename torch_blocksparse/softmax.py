import triton
import torch
import math
from test import sparse_to_dense, dense_to_sparse

fwd_kernels = dict()
fwd_src = '''
__global__ void softmax_fwd(TYPE *X, TYPE scale,
                            int *LUT, TYPE *M,
                            int num_blocks, int sizemax, 
                            int stride_zx, int stride_zm){ 
  int pidhm = get_program_id(0);
  int pidz = get_program_id(1);

  // create index ranges
  int rxm[TM]     = (pidhm*TM + (0 ... TM)) % BLOCK;
  int rbm[TM]     = (pidhm*TM + (0 ... TM)) / BLOCK;
  int rxn[TN]     = (0 ... TN) % BLOCK;
  int rbn[TN]     = (0 ... TN) / BLOCK;

  // extract information from look-up table
  int* header[TM] = LUT + rbm * 2;
  int size[TM]    = *(header + 0);
  int offset[TM]  = *(header + 1);

  // block id and column id
  int blockid[TM, TN] = *(LUT + offset[:, newaxis] + rbn[newaxis, :]);
  int columnid[TM, TN] = *(LUT + offset[:, newaxis] + rbn[newaxis,:] + num_blocks);

  // initialize pointers
  TYPE* pm[TM, TN]  = M + pidz * stride_zm 
                        + columnid * BLOCK
                        + rxn[newaxis, :];

  TYPE* px[TM, TN]  = X + pidz * stride_zx
                        + blockid * BLOCK * BLOCK 
                        + rxm[:,newaxis] * BLOCK 
                        + rxn[newaxis,:];

  // compute fused softmax
  bool check[TM, TN] = rbn[newaxis, :] < size[:, newaxis];
  bool do_mask[TM, TN] = M;
  TYPE m[TM, TN] = (check && do_mask)? *pm : -INFINITY;
  TYPE x[TM, TN] = check ? *px : -INFINITY;
  x *= scale;
  x += (m != 0 && do_mask) ? ((TYPE[TM,TN])-INFINITY) : 0;
  TYPE xmax[TM]  = x[:, max];
  TYPE y[TM, TN] = exp(x - xmax[:, newaxis]);
  TYPE ysum[TM] = (check ? y : 0)[:, +];

  // write-back
  *?(check)px = y / ysum[:, newaxis];
}
'''

bwd_kernels = dict()
bwd_src = '''

__global__ void softmax_bwd(TYPE * X, TYPE scale,
                            TYPE* DX, int* LUT,
                            int sizemax, int stride_zx, int stride_zdx) {
    int pidhm = get_program_id(0);
    int pidz = get_program_id(1);

    // create index ranges
    int rxm[TM] = (pidhm * TM + (0 ... TM)) % BLOCK;
    int rbm[TM] = (pidhm * TM + (0 ... TM)) / BLOCK;
    int rxn[TN] = (0 ... TN) % BLOCK;
    int rbn[TN] = (0 ... TN) / BLOCK;

    // extract information from look-up table
    int* header[TM] = LUT + rbm * 2;
    int size[TM] = *(header + 0);
    int offset[TM] = *(header + 1);

    // initialize pointers to block-sparse input
    int blockid[TM, TN] = *(LUT + offset[:, newaxis] + rbn[newaxis, :]);

    TYPE* px[TM, TN] = X + pidz * stride_zx
                         + blockid * BLOCK * BLOCK
                         + rxm[:, newaxis] * BLOCK
                         + rxn[newaxis, :];

    TYPE* pdx[TM, TN] = DX + pidz * stride_zdx
                           + blockid * BLOCK * BLOCK
                           + rxm[:, newaxis] * BLOCK
                           + rxn[newaxis, :];

    // compute fused softmax backward
    bool check[TM, TN] = rbn[newaxis, :] < size[:, newaxis];
    TYPE x[TM, TN] = check ? *px : 0;
    TYPE dx[TM, TN] = check ? *pdx : 0;
    TYPE xdx[TM, TN] = x * dx;
    TYPE xdxsum[TM] = (check ? xdx : 0)[:, +];
    TYPE y[TM, TN] = x * (dx - xdxsum[:, newaxis]) * scale;

    // write-back
    *? (check)pdx = y;
}
'''

class _sparse_softmax(torch.autograd.Function):

    bwd_kernels = dict()

    @staticmethod
    def make_lut(layout, block):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        # sizes along rows
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h,:,:].sum(-1)))
        # offsets in block format
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        # block indices
        idx = torch.arange(layout.sum())
        # columns
        columns = layout.nonzero()[:, 2]
        # construct look-up table
        offsets += 2*sizes.numel()
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, idx, columns)).type(torch.int32).cuda()
        return lut, sizes.max()

    @staticmethod
    def make_kernel(cache, src, max_k, dtype, block):
        # pad tile to cover the entire reduction
        params = {16384: (1, 32768, 16),
                  8192:  (1, 16384, 16),
                  4096:  (1, 8192, 16),
                  2048:  (1, 4096, 16),
                  1024:  (1, 2048, 16)}
        bound = max(1024, 2**int(math.log2(max_k-1)))
        if bound not in params:
          raise NotImplementedError('Reductions larger than 32768 elements '\
                                    'are not yet implemented')
        TM, TN, num_warps = params[bound]
        # just-in-time compile kernel
        key = (dtype, TM, TN, num_warps)
        if key not in cache:
            defines = {'TM': [TM], 'TN': [TN], 'TYPE': dtype, 'BLOCK': block}
            kernel  = triton.kernel(src, defines=defines, num_warps=[num_warps])
            cache[key] = kernel
        return cache[key]

    @staticmethod
    def forward(ctx, x, scale, layout, block, mask, lut, num_blocks, maxlut, bench, time):
        # run kernel
        kernel = _sparse_softmax.make_kernel(fwd_kernels, fwd_src, maxlut*block, x.dtype, block)
        grid = lambda opt: [triton.cdiv(layout.shape[0] * layout.shape[1] * block, opt.d('TM')),
                            x.shape[0]]
        # handle None mask
        stride_zm = 0 if mask is None else mask.stride(0)
        mask = torch.empty(0, dtype=x.dtype, device=x.device) if mask is None else mask
        # run kernel
        time[0] = kernel(x, scale, lut, mask,\
                         num_blocks, maxlut,\
                         x.stride(0), stride_zm,\
                         grid=grid, bench=bench)
        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, layout, lut)
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        return x
    
    @staticmethod
    def backward(ctx, dx):
        # retrieve from context
        x, layout, lut = ctx.saved_tensors
        block = ctx.block
        maxlut = ctx.maxlut
        scale = ctx.scale
        # run kernel
        kernel = _sparse_softmax.make_kernel(bwd_kernels, bwd_src, maxlut*block, x.dtype, block)
        grid = lambda opt: [triton.cdiv(layout.shape[0] * layout.shape[1] * block, opt.d('TM')),
                            x.shape[0]]
        kernel(x, scale, dx, lut, maxlut, x.stride(0), dx.stride(0), grid=grid)
        return dx, None, None, None, None, None, None, None, None, None

class SparseSoftmax:
    
    def __init__(self, layout, block, scale = 1., mask = None, bench = False):
        self.fwd_lut, self.fwd_maxlut = _sparse_softmax.make_lut(layout, block)
        self.num_blocks = layout.sum()
        self.mask = mask
        self.layout = layout
        self.block = block
        self.bench = bench
        self.scale = scale
    
    def __call__(self, x):
        time_y = [None]
        x = _sparse_softmax.apply(x, self.scale,
                                  self.layout, self.block, self.mask,
                                  self.fwd_lut, self.num_blocks, 
                                  self.fwd_maxlut, self.bench, time_y)
        self.time_y = time_y[0]
        return x

# TODO:
#  - Triton update
#  - Sparse multi-head attention + test code