import triton
import torch
import math

fwd_kernels = dict()
fwd_src = '''
__global__ void softmax_fwd(TYPE *X __readonly __noalias __aligned(16), 
                            float scale,
                            int *LUT __readonly __noalias __aligned(16), 
                            TYPE *RPE __readonly __noalias __aligned(16), 
                            TYPE *KP_M __readonly __noalias __aligned(16), 
                            TYPE *ATTN_M __readonly __noalias __aligned(16),
                            int num_blocks, int num_blocks_per_head,
                            int sizemax, 
                            long stride_zx __multipleof(BLOCK),
                            long stride_zrpe __multipleof(BLOCK),
                            int stride_hrpe __multipleof(BLOCK),
                            int stride_srpe __multipleof(BLOCK),
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
#ifdef APPLY_RPE
  // pointers to relative position embedding
  TYPE* prpe[TN] = RPE + pidz * stride_zrpe
                            + blockid / num_blocks_per_head * stride_hrpe
                            + columnid * BLOCK
                            + rowid * BLOCK * stride_srpe
                            + rxm * stride_srpe
                            + rxn;
#endif

#ifdef APPLY_KP_MASK
  // pointers to key padding mask
  TYPE* pkp_m[TN]  = KP_M + pidz * stride_zkpm 
                          + columnid * BLOCK
                          + rxn;
#endif

#ifdef APPLY_ATTN_MASK
  // pointers to attention mask
  TYPE* pattn_m[TN] = ATTN_M + columnid * BLOCK 
                             + rowid * BLOCK * stride_zattnm
                             + rxm * stride_zattnm
                             + rxn;
#endif
 
  // load  input
  TYPE x[TN] =  check ? *px : -INFINITY;

#ifdef APPLY_RPE
  // load relative position embedding
  TYPE rpe[TN] = check ? *prpe : 0;
#endif

#ifdef APPLY_KP_MASK
  // load key-padding mask
  TYPE kp_m[TN] = check ? *pkp_m : -INFINITY;
#endif

#ifdef APPLY_ATTN_MASK
  // load attention mask
  TYPE attn_m[TN] = check ? *pattn_m : -INFINITY;
#endif

  // compute softmax in float
#ifdef APPLY_RPE
  float Frpe[TN] = rpe;
#endif

#ifdef APPLY_KP_MASK
  float Fkp_m[TN] = kp_m;
#endif

#ifdef APPLY_ATTN_MASK
  float Fattn_m[TN] = attn_m;
#endif

#ifdef KP_MASK_MUL
  Fkp_m = (Fkp_m == 0) ? (float[TN])-INFINITY : 0;
#endif

#ifdef ATTN_MASK_MUL
  Fattn_m = (Fattn_m == 0) ? (float[TN])-INFINITY : 0;
#endif

  float Fx[TN] = x;

#ifdef APPLY_SCALE
  Fx = Fx * scale; // apply scale
#endif

#ifdef APPLY_RPE
  Fx = Fx + Frpe; // apply relative position embedding
#endif

#ifdef APPLY_KP_MASK
  Fx = Fx + Fkp_m; // apply key padding mask
#endif

#ifdef APPLY_ATTN_MASK
  Fx = Fx + Fattn_m; // apply attention mask
#endif

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
    def make_lut(layout, block, device):
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
        lut = torch.cat((header, core)).type(torch.int32).to(device)
        return lut, int(sizes.max())

    @staticmethod
    def make_kernel(cache, src, max_k, dtype, block, apply_scale, apply_rpe, apply_kp_mask, apply_attn_mask, kp_mask_mode, attn_mask_mode):
        if max_k >= 32768:
          raise NotImplementedError('Reductions larger than 32768 elements '\
                                    'are not yet implemented')
        num_warps = 4 if max_k < 512 else (8 if max_k < 2048 else 16)
        pad = num_warps * 32 * 2
        TN = (int(max_k) + pad-1)//pad * pad
        # just-in-time compile kernel
        key = (block, dtype, num_warps, TN, apply_scale, apply_rpe, apply_kp_mask, apply_attn_mask, kp_mask_mode, attn_mask_mode)
        if key not in cache:
            defines = {'TM': [1], 'TN': [TN], 'TYPE': dtype, 'BLOCK': block,
                       'INFINITY': {torch.float32: 'F32_INFINITY',
                                    torch.float16: 'F16_INFINITY'}[dtype]}
            if apply_scale:
                defines['APPLY_SCALE'] = True
            if apply_rpe:
                defines['APPLY_RPE'] = True
            if apply_kp_mask:
                defines['APPLY_KP_MASK'] = True
                if kp_mask_mode == 'mul':
                    defines['KP_MASK_MUL'] = True
            if apply_attn_mask:
                defines['APPLY_ATTN_MASK'] = True
                if attn_mask_mode == 'mul':
                    defines['ATTN_MASK_MUL'] = True
            kernel  = triton.kernel(src, defines=defines, num_warps=[num_warps])
            cache[key] = kernel
        return cache[key]

    @staticmethod
    def forward(ctx, x, scale, rpe, key_padding_mask, attn_mask, kp_mask_mode, attn_mask_mode,
                spdims, block, lut, num_blocks, num_blocks_per_head, maxlut, bench, time):
        apply_scale = False if scale == 1.0 else True

        # handle None rpe
        if rpe is None:
            apply_rpe = False
            stride_zrpe, stride_hrpe, stride_srpe = 0, 0, 0
            rpe = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_rpe = True
            stride_zrpe, stride_hrpe, stride_srpe = rpe.stride(0), rpe.stride(1), rpe.stride(2)

        # handle None key_padding_mask
        if key_padding_mask is None:
            apply_kp_mask = False
            stride_zkpm = 0
            key_padding_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_kp_mask = True
            stride_zkpm = key_padding_mask.stride(0)

        # handle None attention_mask
        if attn_mask is None:
            apply_attn_mask = False
            stride_zattnm = 0
            attn_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_attn_mask = True
            stride_zattnm = attn_mask.stride(0)


        # run kernel
        kernel = _sparse_softmax.make_kernel(fwd_kernels, fwd_src, maxlut*block, x.dtype, block,
                                            apply_scale, apply_rpe, apply_kp_mask, apply_attn_mask,
                                            kp_mask_mode, attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [triton.cdiv(spdims[0] * spdims[1] * block, opt.d('TM')), M]

        # run kernel
        time[0] = kernel(x, scale, lut, rpe, key_padding_mask, attn_mask,\
                         num_blocks, num_blocks_per_head, maxlut,\
                         x.stride(0),\
                         stride_zrpe, stride_hrpe, stride_srpe,\
                         stride_zkpm, stride_zattnm,\
                         grid=grid, bench=bench)
        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.apply_scale = apply_scale
        ctx.apply_rpe = apply_rpe
        ctx.apply_kp_mask = apply_kp_mask
        ctx.apply_attn_mask = apply_attn_mask
        ctx.kp_mask_mode = kp_mask_mode
        ctx.attn_mask_mode = attn_mask_mode
        return x
    
    @staticmethod
    def backward(ctx, dx):
        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        kernel = _sparse_softmax.make_kernel(bwd_kernels, bwd_src, ctx.maxlut*ctx.block, x.dtype, ctx.block, 
                                            ctx.apply_scale, ctx.apply_rpe, ctx.apply_kp_mask, ctx.apply_attn_mask,
                                            ctx.kp_mask_mode, ctx.attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [triton.cdiv(ctx.spdims[0] * ctx.spdims[1] * ctx.block, opt.d('TM')), M]
        kernel(x, ctx.scale, dx, lut, ctx.maxlut, x.stride(0), dx.stride(0), grid=grid)
        return dx, None, None, None, None, None, None, None, None, None, None, None, None, None, None

class Softmax:
    
    sparse_softmax = _sparse_softmax.apply

    def make_lut(self, device):
        key = (device, )
        if key not in self.lut_cache:
          self.lut_cache[key] = _sparse_softmax.make_lut(self.layout, self.block, device)
        return self.lut_cache[key]

    def __init__(self, layout, block, bench = False):
        self.num_blocks = layout.sum()
        self.num_blocks_per_head = self.num_blocks / layout.shape[0]
        self.spdims = layout.shape
        self.layout = layout
        self.block = block
        self.bench = bench
        self.lut_cache = dict()
    
    def __call__(self, x, scale = 1., rpe = None, key_padding_mask = None, attn_mask = None,
            key_padding_mask_mode='add', attn_mask_mode='add'):
        time_y = [None]
        if rpe is not None and rpe.dtype != x.dtype:
            raise ValueError('relative position embedding must be %s' % x.dtype)
        if attn_mask is not None and attn_mask.dtype != x.dtype:
            raise ValueError('Attention mask must be %s' % x.dtype)
        if key_padding_mask is not None and key_padding_mask.dtype != x.dtype:
            raise ValueError('Key padding mask must be %s' % x.dtype)
        lut, maxlut = self.make_lut(x.device)
        x = Softmax.sparse_softmax(x, scale, rpe, key_padding_mask, attn_mask,
                                  key_padding_mask_mode, attn_mask_mode,
                                  self.spdims, self.block,
                                  lut, self.num_blocks, self.num_blocks_per_head,
                                  maxlut, self.bench, time_y)
        self.time_y = time_y[0]
        return x

# TODO:
#  - Sparse multi-head attention + test code
