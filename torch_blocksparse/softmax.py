import triton
import torch
import math

fwd_kernels = dict()
fwd_src = '''
__global__ void softmax_fwd(TYPE *X __readonly __noalias __aligned(16), 
                            float scale,
                            int *LUT __readonly __noalias __aligned(16), 
                            TYPE *KP_M __readonly __noalias __aligned(16), 
                            TYPE *ATTN_M __readonly __noalias __aligned(16),
                            int num_blocks, 
                            int sizemax, 
                            long stride_zx __multipleof(BLOCK), 
                            int stride_zkpm __multipleof(BLOCK), 
                            int stride_zattnm __multipleof(BLOCK)){ 
  int pidhm = get_program_id(0);
  int pidz = get_program_id(1);

  // create index ranges
  int rxm     = pidhm % BLOCK;
  int rbm     = pidhm / BLOCK;
  int rxn[TN] = (0 ... TN) % BLOCK;
  int rbn[TN] = (0 ... TN) / BLOCK;

  // extract information from look-up table
  int* header = LUT + rbm * 2;
  int size    = *(header + 0);
  int offset  = *(header + 1);

  bool check[TN] = rbn < size;
  int   rbmn[TN] = check ? rbn : size - 1;

  // block id and column id
  long blockid [TN]  = *(LUT + offset + rbmn*3 + 0);
  long columnid[TN]  = *(LUT + offset + rbmn*3 + 1);
  long rowid   [TN]  = *(LUT + offset + rbmn*3 + 2);

  // pointers to X
  TYPE* px[TN]  = X + pidz * stride_zx
                    + blockid * BLOCK * BLOCK 
                    + rxm * BLOCK 
                    + rxn;

  // pointers to key padding mask
  TYPE* pkp_m[TN]  = KP_M + pidz * stride_zkpm 
                          + columnid * BLOCK
                          + rxn;

  // pointers to attention mask
  TYPE* pattn_m[TN] = ATTN_M + columnid * BLOCK 
                             + rowid * BLOCK * stride_zattnm
                             + rxm * stride_zattnm
                             + rxn;

 
  // load  input
  TYPE x[TN] =  check ? *px : -INFINITY;
  // load key-padding mask
  bool do_kp_mask[TN] = KP_M;
  TYPE kp_m[TN] = (check && do_kp_mask)? *pkp_m : -INFINITY;
  // load attention mask
  bool do_attn_mask[TN] = ATTN_M;
  TYPE attn_m[TN] = (check && do_attn_mask)? *pattn_m : -INFINITY;

  // compute softmax in float
  float Fkp_m[TN] = kp_m;
  float Fattn_m[TN] = attn_m;
#ifdef KP_MASK_MUL
  Fkp_m = (Fkp_m == 0) ? (float[TN])-INFINITY : 0;
#endif
#ifdef ATTN_MASK_MUL
  Fattn_m = (Fattn_m == 0) ? (float[TN])-INFINITY : 0;
#endif
  float Fx[TN] = x;
  Fx = Fx * scale; // apply scale
  Fx = Fx + (do_kp_mask ? Fkp_m : 0); // apply key padding mask
  Fx = Fx + (do_attn_mask ? Fattn_m : 0); // apply attention mask
  float Fxmax  = Fx[max];
  float Fy[TN] = exp(Fx - Fxmax);
  float Fysum = (check ? Fy : 0)[+];

  // write-back in half/float
  TYPE y[TN] = Fy;
  TYPE ysum = Fysum;
  *?(check)px = y / ysum;
}
'''

bwd_kernels = dict()
bwd_src = '''

__global__ void softmax_bwd(TYPE * X __readonly __noalias __aligned(16), 
                            float scale,
                            TYPE* DX __readonly __noalias __aligned(16), 
                            int* LUT,
                            int sizemax, 
                            long stride_zx __multipleof(BLOCK), 
                            long stride_zdx __multipleof(BLOCK)) {
    int pidhm = get_program_id(0);
    int pidz = get_program_id(1);

    // create index ranges
    int rxm = pidhm % BLOCK;
    int rbm = pidhm / BLOCK;
    int rxn[TN] = (0 ... TN) % BLOCK;
    int rbn[TN] = (0 ... TN) / BLOCK;

    // extract information from look-up table
    int* header = LUT + rbm * 2;
    int size    = *(header + 0);
    int offset  = *(header + 1);

    // bounds checking on lut
    bool check[TN] = rbn < size;
    int rbmn[TN] = check ? rbn : size - 1;

    // initialize pointers to block-sparse input
    long blockid[TN] = *(LUT + offset + rbmn*3);

    TYPE* px[TN] = X + pidz * stride_zx
                         + blockid * BLOCK * BLOCK
                         + rxm * BLOCK
                         + rxn;

    TYPE* pdx[TN] = DX + pidz * stride_zdx
                           + blockid * BLOCK * BLOCK
                           + rxm * BLOCK
                           + rxn;

    // compute fused softmax backward
    TYPE x[TN] = check ? *px : 0;
    TYPE dx[TN] = check ? *pdx : 0;
    float Fdx[TN] = dx;
    float Fx[TN] = x;
    float Fxdx[TN] = Fdx*Fx;
    float Fxdxsum = Fxdx[+];
    float Fy[TN] = Fx * (Fdx - Fxdxsum) * scale;
    TYPE y[TN] = Fy;

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
        rows = layout.nonzero()[:, 1]
        columns = layout.nonzero()[:, 2]
        core   = torch.stack((idx, columns, rows), dim=1).view(-1)
        # construct look-up table
        offsets = offsets*3 + 2*sizes.numel()
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, core)).type(torch.int32).cuda()
        return lut, int(sizes.max())

    @staticmethod
    def make_kernel(cache, src, max_k, dtype, block, kp_mask_mode, attn_mask_mode):
        if max_k >= 32768:
          raise NotImplementedError('Reductions larger than 32768 elements '\
                                    'are not yet implemented')
        num_warps = 4 if max_k < 512 else (4 if max_k < 2048 else 8)
        pad = num_warps * 32 * 2
        TN = (int(max_k) + pad-1)//pad * pad
        # just-in-time compile kernel
        key = (block, dtype, num_warps, TN, kp_mask_mode, attn_mask_mode)
        if key not in cache:
            defines = {'TM': [1], 'TN': [TN], 'TYPE': dtype, 'BLOCK': block,
                       'INFINITY': {torch.float32: 'F32_INFINITY',
                                    torch.float16: 'F16_INFINITY'}[dtype]}
            if kp_mask_mode == 'mul':
                defines['KP_MASK_MUL'] = True
            if attn_mask_mode == 'mul':
                defines['ATTN_MASK_MUL'] = True
            kernel  = triton.kernel(src, defines=defines, num_warps=[num_warps])
            cache[key] = kernel
        return cache[key]

    @staticmethod
    def forward(ctx, x, scale, key_padding_mask, attn_mask, kp_mask_mode, attn_mask_mode,
                spdims, block, lut, num_blocks, maxlut, bench, time):
        # run kernel
        kernel = _sparse_softmax.make_kernel(fwd_kernels, fwd_src, maxlut*block, x.dtype, block, 
                                             kp_mask_mode, attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [triton.cdiv(spdims[0] * spdims[1] * block, opt.d('TM')), M]
        # handle None key_padding_mask
        stride_zkpm = 0 if key_padding_mask is None else key_padding_mask.stride(0)
        key_padding_mask = torch.empty(0, dtype=x.dtype, device=x.device) if key_padding_mask is None else key_padding_mask
        # handle None attention_mask
        stride_zattnm = 0 if attn_mask is None else attn_mask.stride(0)
        attn_mask = torch.empty(0, dtype=x.dtype, device=x.device) if attn_mask is None else attn_mask
        # run kernel
        time[0] = kernel(x, scale, lut, key_padding_mask, attn_mask,\
                         num_blocks, maxlut,\
                         x.stride(0), stride_zkpm, stride_zattnm,\
                         grid=grid, bench=bench)
        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.kp_mask_mode = kp_mask_mode
        ctx.attn_mask_mode = attn_mask_mode
        return x
    
    @staticmethod
    def backward(ctx, dx):
        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        kernel = _sparse_softmax.make_kernel(bwd_kernels, bwd_src, ctx.maxlut*ctx.block, x.dtype, ctx.block, 
                                             ctx.kp_mask_mode, ctx.attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [triton.cdiv(ctx.spdims[0] * ctx.spdims[1] * ctx.block, opt.d('TM')), M]
        kernel(x, ctx.scale, dx, lut, ctx.maxlut, x.stride(0), dx.stride(0), grid=grid)
        return dx, None, None, None, None, None, None, None, None, None, None, None, None

class Softmax:
    
    sparse_softmax = _sparse_softmax.apply

    def __init__(self, layout, block, bench = False):
        self.fwd_lut, self.fwd_maxlut = _sparse_softmax.make_lut(layout, block)
        self.num_blocks = layout.sum()
        self.spdims = layout.shape
        self.block = block
        self.bench = bench
    
    def __call__(self, x, scale = 1., key_padding_mask = None, attn_mask = None, 
                 key_padding_mask_mode='add', attn_mask_mode='add'):
        time_y = [None]
        if attn_mask is not None and attn_mask.dtype != x.dtype:
            raise ValueError('Attention mask must be %s' % x.dtype)
        if key_padding_mask is not None and key_padding_mask.dtype != x.dtype:
            raise ValueError('Key padding mask must be %s' % x.dtype)
        x = Softmax.sparse_softmax(x, scale, key_padding_mask, attn_mask, 
                                  key_padding_mask_mode, attn_mask_mode,
                                  self.spdims, self.block,
                                  self.fwd_lut, self.num_blocks, 
                                  self.fwd_maxlut, self.bench, time_y)
        self.time_y = time_y[0]
        return x

# TODO:
#  - Sparse multi-head attention + test code
