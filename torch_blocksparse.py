import triton
import torch
import math
from collections import namedtuple

class _linear(torch.autograd.Function):

  src = '''
  __global__ void NAME (TYPE* A __readonly  __noalias __aligned(16),
                       TYPE* B __readonly  __noalias __aligned(16),
                       TYPE* C __noalias __aligned(16),
                       int lda __multipleof(8), 
                       int ldb __multipleof(8), 
                       int ldc __multipleof(8),
                       int M, int Kmax,
                       int* lut,
                       int* locks, int nlocks) {
    /* ---------------- */
    /*    Prologue      */
    /* ---------------- */
    // program ids
    int pid0 = get_program_id(0);
    int pid1 = get_program_id(1);
#ifdef DW
    // load LUT header
    int *header = lut + pid1 * 2;
    int i = *(header + 0);
    int j = *(header + 1);
    int K = Kmax / TZ;
    int lockid = select(TZ > 1, 1, 0);
    int offka = pid0 * K;
    int offkb = pid0 * K;
    int offm = i * TM;
    int offn = j * TN;
    int maxid = get_num_programs(0);
#else
    // load LUT header
    int *header = lut + pid0 * 5;
    int offset = *(header + 0);
    int K      = *(header + 1);
    int column = *(header + 2);
    int lockid = *(header + 3);
    int maxid = *(header + 4);
    int *pinc   = lut + offset;
    int offka __multipleof(8) = *pinc;
    int offkb __multipleof(8) = *(pinc + 1);
    int offm = pid1 * TM;
    int offn = column * TN;
#endif
    // initialize a, b pointers
    int rka[TK] = offka + 0 ... TK;
    int ram[TM] = offm + (0 ... TM);
#ifdef DW
    int rkb[TK] = offkb + 0 ... TK;
    int rbn[TN] = offn + (0 ... TN);
    TYPE* pa[TM, TK] = A + ram[:, newaxis] * STRIDE_AM + rka[newaxis, :] * STRIDE_AK;
    TYPE* pb[TK, TN] = B + rbn[newaxis, :] * STRIDE_BN + rkb[:, newaxis] * STRIDE_BK;
#else
    int rkb[TK] = 0 ... TK;
    int rbn[TN] = 0 + (0 ... TN);
    TYPE* pa[TM, TK] = A + ram[:, newaxis] * STRIDE_AM + rka[newaxis, :] * STRIDE_AK;
    TYPE* pb[TK, TN] = B + offkb + rbn[newaxis, :] * STRIDE_BN + rkb[:, newaxis] * STRIDE_BK;
#endif
    // pre-fetch
    bool checka[TM, TK] = (ram[:, newaxis] < M);
    bool checkb[TK, TN] = K > 0;
    TYPE a[TM, TK] = checka ? *pa : 0;
    TYPE b[TK, TN] = checkb ? *pb : 0;

    /* ---------------- */
    /*    Inner Loop    */
    /* ---------------- */
    // create result tile
    float acc[TM, TN] = 0;
#ifdef DW
    int step = TK;
#else
    int step = 1;
#endif
    for(int k = K; k > 0; k -= step) {
      acc += a @ b;
      // update pointers
#ifdef DW
      int inc_a = TK * STRIDE_AK;
      int inc_b = TK * STRIDE_BK;
#else
      pinc += 2;
      int inc_a __multipleof(8) = *pinc;
      int inc_b __multipleof(8) = *(pinc + 1);
      //TODO: __multipleof is ignored when * STRIDE_AK is inline above!
      inc_a = inc_a * STRIDE_AK;
#endif
      pa += inc_a;
      pb += inc_b;
      // pre-fetch
      bool checka[TM, TK] = k > 1;
      bool checkb[TK, TN] = k > 1;
      a = *?(checka)pa;
      b = *?(checkb)pb;
    }
    TYPE c[TM, TN] = acc;

    /* ---------------- */
    /*    Epilogue      */
    /* ---------------- */
    // initialize c pointers
#ifdef DW
    int   rcm[TM]    = (0 ... TM);
    int   rcn[TN]    = (0 ... TN);
    TYPE* pc[TM, TN] = C + rcm[:, newaxis]*TN + rcn[newaxis, :] + pid1*TM*TN;
    bool checkc[TM, TN] = 1;
#else
    int   rcm[TM]    = offm + (0 ... TM);
    int   rcn[TN]    = offn + (0 ... TN);
    TYPE* pc[TM, TN] = C + rcm[:, newaxis]*ldc + rcn[newaxis, :];
    bool  checkc[TM, TN] = rcm[:, newaxis] < M;
#endif
    // write-back directly
    if(lockid == 0) {
      *?(checkc) pc = c;
    }
    // accumulate partial result using spin-locks
    else {
      int *plock = locks + get_program_id(1)*nlocks + lockid - 1;
      int *pcount = plock + get_num_programs(1)*nlocks;
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % maxid);
      atomic_xchg(plock, 0);
    }
  }
'''

  # dictionaries for cached triton kernels
  y_kernel = dict()
  dx_kernel = dict()
  dw_kernel = dict()

  # Given an array sizes representing reduction size for each
  # column of a block-sparse matrix multiplication,
  # performs load-balancing to achieve more smaller reductions
  # of size seg_size
  @staticmethod
  def load_balance(sizes, block_size):
    # segment size
    # heuristics taken from OpenAI blocksparse code
    # https://github.com/openai/blocksparse/blob/master/blocksparse/matmul.py#L95
    max_size = sizes.max()
    min_size = sizes[torch.nonzero(sizes)].min()
    if max_size > min_size > 2.0:
      seg_max = max(triton.cdiv(max_size, 4), min_size*2)
    else:
      seg_max = max_size
    seg_min = max(triton.cdiv(seg_max, 4), 4)
    # split reduction into segments
    div = sizes // seg_max
    rem = sizes % seg_max
    packs = div + (sizes < seg_min).long() + (rem >= seg_min).long()
    width = packs.sum()
    segments = torch.empty(width, dtype=sizes.dtype)
    column = torch.empty_like(segments)
    lockid = torch.zeros_like(segments)
    maxid = torch.zeros_like(segments)
    nlocks = 0
    current = 0
    col_idx = 0
    for i in range(len(sizes)):
      d, r = div[i], rem[i]
      isempty = sizes[i] < seg_min
      last = current + d + (r >= seg_min) + isempty
      # column id
      column[current:last] = col_idx
      # lock id
      if d > 1 or (d == 1 and r >= seg_min):
        nlocks += 1
        lockid[current:last] = nlocks
        maxid[current:last] = last - current
      # segment size
      segments[current:current+d] = seg_max
      if r < seg_min and not isempty:
        segments[current+d-1] += r
      if r >= seg_min or isempty:
        segments[current+d] = r
      current = last
      col_idx += 1
    offsets = torch.zeros_like(segments)
    offsets[1:] = torch.cumsum(segments[:-1], dim=0)
    return segments, column, lockid, maxid, offsets

  # Given a binary mask of 0s and 1s,
  # Construct look-up table for efficient execution on GPUs
  @staticmethod
  def make_ydx_lut(mask, block_size, step, trans):
    # nonzeros
    if trans:
      sizes = torch.sum(mask, 1)
      nnz = torch.nonzero(mask)
    else:
      sizes = torch.sum(mask, 0)
      nnz = torch.nonzero(mask.T)
    num_blocks = nnz.size(0)
    # load-balancing
    segments, column, lockid, maxid, offsets = _linear.load_balance(sizes, block_size)
    # pointer increments
    offsets = torch.min(offsets, (num_blocks - 1)*torch.ones_like(offsets))
    idx = nnz[:, 1]*block_size
    xincs = idx.clone() 
    xincs[1:] -= idx[:-1]
    # divide block_size into multiple steps
    div = block_size // step
    xincs = xincs.view(-1, 1).repeat(1, div)
    xincs[:, 1:] = step
    xincs[:, 0 ] -= (div-1)*step
    # first increment for each reduction is actually the offset
    xincs[offsets[segments>0], 0] = idx[offsets[segments>0]]
    xincs = xincs.view(-1)
    # block-sparse input increments
    if trans:
      widx = torch.arange(num_blocks)
    else:
      maskw = mask.clone()
      maskw[maskw > 0] = 1 + torch.arange(maskw.sum())
      widx = maskw.T[maskw.T > 0] - 1
    widx = widx * block_size * block_size
    wincs = widx.clone()
    wincs[1:] -= widx[:-1]
    wincs = wincs.view(-1, 1).repeat(1, div)
    if trans:
      wincs[:, 1:] = step
      wincs[:, 0] -= (div-1)*step
    else:
      wincs[:, 1:] = step*block_size
      wincs[:, 0] -= (div - 1)*step*block_size
    wincs[offsets[segments>0], 0] = widx[offsets[segments>0]]
    wincs = wincs.view(-1)
    # adjust offset and segment size
    offsets *= 2*div
    segments *= div
    # create header
    width = column.size(0)
    offsets += 5*width
    header = torch.stack((offsets, segments, column, lockid, maxid), dim=1).view(-1).contiguous()
    incs = torch.stack((xincs, wincs), dim=1).view(-1).contiguous()
    # create lut
    lut = torch.cat((header, incs)).type(torch.int32).cuda()
    # create locks
    num_locks = max(1, lockid.max())
    locks = torch.zeros((2*width, num_locks), dtype=torch.int32).cuda()
    return lut, locks, width
  
  @staticmethod
  def make_dw_lut(mask, block_size):
    nnz = torch.nonzero(mask)
    # create lut
    width = nnz.size(0)
    i = nnz[:, 0]
    j = nnz[:, 1]
    lut = torch.stack((i, j), dim=1).view(-1).contiguous()
    lut = lut.type(torch.int32).cuda()
    # create locks
    num_locks = 1
    locks = torch.zeros((2*width, num_locks), dtype=torch.int32).cuda()
    return lut, locks, width

  @staticmethod
  def forward(ctx, x, w, mask, block_size, 
              y_lut, y_locks, y_width,
              dx_lut, dx_locks, dx_width,
              dw_lut, dw_locks, dw_width,
              bench_y, bench_dx, bench_dw,
              timings):
    M, Kx = x.size()
    N = block_size * mask.size(1)
    dtype = x.dtype
    # memory strides
    lda = Kx
    ldb = N
    ldc = N
    # create kernel
    key = (dtype, block_size)
    if key not in _linear.y_kernel:
      defines = {'TM': 128, 'TN': block_size, 'TK': 8, 'TYPE': dtype,
                'STRIDE_AM': 'lda', 'STRIDE_AK': '1',
                'STRIDE_BN': '1', 'STRIDE_BK': block_size,
                'NAME': 'y_kernel'}
      _linear.y_kernel[key] = triton.kernel(_linear.src, defines=defines)
    kernel = _linear.y_kernel[key]
    # allocate output
    y = torch.empty((M, N), dtype=dtype, device=x.device)
    # launch kernel
    grid = lambda opt: [y_width, triton.cdiv(M, opt.d('TM'))]
    timings.ty = kernel(x, w, y, lda, ldb, ldc, M, Kx, y_lut, y_locks, y_locks.size(1), grid=grid, bench=bench_y)
    # save information in context
    ctx.dx_width = dx_width
    ctx.dw_width = dw_width
    ctx.kernel = kernel
    ctx.block_size = block_size
    ctx.bench_dx = bench_dx
    ctx.bench_dw = bench_dw
    ctx.timings = timings
    ctx.save_for_backward(x, w, dx_lut, dx_locks, dw_lut, dw_locks)
    return y
  
  @staticmethod
  def backward(ctx, dy):
    # retrieve information in context
    x, w, dx_lut, dx_locks, dw_lut, dw_locks = ctx.saved_tensors
    dx_width = ctx.dx_width
    dw_width = ctx.dw_width
    block_size = ctx.block_size
    kernel = ctx.kernel
    bench_dx = ctx.bench_dx
    bench_dw = ctx.bench_dw
    timings = ctx.timings
    # shapes
    M, N = dy.size()
    _, K = x.size()
    dtype = x.dtype
    ################
    # input gradient
    ################
    dx = None
    if ctx.needs_input_grad[0]:
      # create kernel
      key = (dtype, block_size)
      if key not in _linear.dx_kernel:
        defines =  {'TM': 128, 'TN': block_size, 'TK': 8, 'TYPE': dtype,
                    'STRIDE_AM': 'lda', 'STRIDE_AK': '1',
                    'STRIDE_BN': block_size, 'STRIDE_BK': '1',
                    'NAME': 'dx_kernel'}
        _linear.dx_kernel[key] = triton.kernel(_linear.src, defines=defines)
      kernel = _linear.dx_kernel[key]
      # allocate output
      dx = torch.empty_like(x)
      # launch kernel
      grid = lambda opt: [dx_width, triton.cdiv(M, opt.d('TM'))]
      timings.tdx = kernel(dy, w, dx, N, N, K, M, N, dx_lut, dx_locks, dx_locks.size(1), grid=grid, bench=bench_dx)
    #################
    # weight gradient
    #################
    dw = None
    if ctx.needs_input_grad[1]:
      # create kernel
      key = (dtype, block_size)
      if key not in _linear.dw_kernel:
        defines =  {'TM': block_size, 'TN': block_size, 'TK': 16, 'TYPE': dtype,
                    'STRIDE_AM': '1', 'STRIDE_AK': 'lda',
                    'STRIDE_BN': '1', 'STRIDE_BK': 'ldb',
                    'DW': True, 'TZ': 2,
                    'NAME': 'dw_kernel'}
        _linear.dw_kernel[key] = triton.kernel(_linear.src, defines=defines)
      kernel = _linear.dw_kernel[key]
      # allocate output
      dw = torch.zeros_like(w)
      # launch kernel
      grid = lambda opt: [opt.d('TZ'), dw_width]
      timings.tdw = kernel(x, dy, dw, K, N, N, K, M, dw_lut, dw_locks, dw_locks.size(1), grid=grid, bench=bench_dw)
    # done
    return dx, dw, None, None,\
          None, None, None,\
          None, None, None,\
          None, None, None,\
          None, None, None,\
          None
linear = _linear.apply

class Linear(torch.nn.Module):

  def __init__(self, in_features, out_features, block_size, mask):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.block_size = block_size
    self.mask = mask
    self.weight = torch.nn.Parameter(torch.Tensor(mask.sum(), block_size, block_size))
    self.reset_parameters()
    # benchmark
    self.bench_y = 0
    self.bench_dx = 0
    self.bench_dw = 0
    # timings
    self.timings = namedtuple('Timings', 'ty tdx tdw')
    # create look-up tables
    self.y_lut, self.y_locks, self.y_width = _linear.make_ydx_lut(mask, block_size, 8, False)
    self.dx_lut, self.dx_locks, self.dx_width = _linear.make_ydx_lut(mask, block_size, 8, True)
    self.dw_lut, self.dw_locks, self.dw_width = _linear.make_dw_lut(mask, block_size)
  
  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #if self.bias is not None:
    #    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
    #    bound = 1 / math.sqrt(fan_in)
    #    torch.nn.init.uniform_(self.bias, -bound, bound)
    
  def forward(self, input):
    return linear(input, self.weight, self.mask, self.block_size,
                  self.y_lut, self.y_locks, self.y_width,
                  self.dx_lut, self.dx_locks, self.dx_width,
                  self.dw_lut, self.dw_locks, self.dw_width,
                  self.bench_y, self.bench_dx, self.bench_dw,
                  self.timings)