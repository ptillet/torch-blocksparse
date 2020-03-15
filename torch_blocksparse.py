import triton
import torch
import math
from collections import namedtuple

src = '''
  __global__ void NAME (TYPE* A __readonly  __noalias __aligned(16),
                        TYPE* B __readonly  __noalias __aligned(16),
                        TYPE* C __noalias __aligned(16),
                        int lda __multipleof(8), 
                        int ldb __multipleof(8), 
                        int ldc __multipleof(8),
                        int DS0,
                        int Kmax,
                        int* lut,
                        int* locks, int nlocks) {
    /* ---------------- */
    /*    Prologue      */
    /* ---------------- */
    // program ids
    int pid0 = get_program_id(0);
    int pid1 = get_program_id(1);
#ifdef SDD
    // load LUT header
    int *header = lut + pid1 * 2;
    int i = *(header + 0);
    int j = *(header + 1);
    int AS1 = Kmax / TZ;
    int lockid = select(TZ > 1, 1, 0);
    int offka = pid0 * AS1;
    int offkb = pid0 * AS1;
    int offma = i * TM;
    int offnb = j * TN;
    int offmc = 0;
    int offnc = 0;
    int offza = 0;
    int offzb = 0;
    int offzc = pid1 * TM * TN;
    int maxid = get_num_programs(0);
#else
    // load LUT header
    int *header = lut + pid0 * 5;
    int offset = *(header + 0);
    int AS1    = *(header + 1);
    int column = *(header + 2);
    int lockid = *(header + 3);
    int maxid  = *(header + 4);
    int *pinc  = lut + offset;
#ifdef DSD
    // output offset
    int offnc = pid1 * TN;
    int offmc = column * TM;
    int offzc = 0;
    // dense input offset
    int offnb = pid1 * TN;
    int offkb __multipleof(8) = *pinc;
    int offzb = 0;
    // sparse input offset
    int offma = 0;
    int offka = 0;
    int offza __multipleof(8) = *(pinc + 1);
#endif
#ifdef DDS
    // output offset
    int offmc = pid1 * TM;
    int offnc = column * TN;
    int offzc = 0;
    // dense input offset
    int offma = pid1 * TM;
    int offka __multipleof(8) = *pinc;
    int offza = 0;
    // sparse input offset
    int offnb = 0;
    int offkb = 0;
    int offzb __multipleof(8) = *(pinc + 1);
#endif
#endif
    // initialize a, b pointers
    int rka[TK] = offka + 0 ... TK;
    int rkb[TK] = offkb + 0 ... TK;
    int ram[TM] = offma + 0 ... TM;
    int rbn[TN] = offnb + 0 ... TN;
    TYPE* pa[TM, TK] = A + offza + ram[:, newaxis] * STRIDE_AM + rka[newaxis, :] * STRIDE_AK;
    TYPE* pb[TK, TN] = B + offzb + rbn[newaxis, :] * STRIDE_BN + rkb[:, newaxis] * STRIDE_BK;
    // pre-fetch
    bool checka[TM, TK] = ram[:, newaxis] < DS0;
    bool checkb[TK, TN] = AS1 > 0;
    TYPE a[TM, TK] = checka ? *pa : 0;
    TYPE b[TK, TN] = checkb ? *pb : 0;

    /* ---------------- */
    /*    Inner Loop    */
    /* ---------------- */
    // create result tile
    float acc[TM, TN] = 0;
    int step = TK;
    for(int k = AS1; k > 0; k -= step) {
      acc += a @ b;
      // update pointers
#ifdef SDD
      int inc_a = TK * STRIDE_AK;
      int inc_b = TK * STRIDE_BK;
#else
      pinc += 2;
#ifdef DSD
      int inc_b __multipleof(8) = *pinc;
      int inc_a __multipleof(8) = *(pinc + 1);
      inc_b = inc_b * STRIDE_BK;
#endif
#ifdef DDS
      int inc_a __multipleof(8) = *pinc;
      int inc_b __multipleof(8) = *(pinc + 1);
      inc_a = inc_a * STRIDE_AK;
#endif
#endif
      pa += inc_a;
      pb += inc_b;
      // pre-fetch
      bool checka[TM, TK] = k > TK;
      bool checkb[TK, TN] = k > TK;
      a = *?(checka)pa;
      b = *?(checkb)pb;
    }
    TYPE c[TM, TN] = acc;

    /* ---------------- */
    /*    Epilogue      */
    /* ---------------- */
    // initialize c pointers
    int   rcm[TM]    = offmc + 0 ... TM;
    int   rcn[TN]    = offnc + 0 ... TN;
    bool  checkc[TM, TN] = rcm[:, newaxis] < DS0;
    TYPE* pc[TM, TN] = C + offzc + rcm[:, newaxis]*STRIDE_CM + rcn[newaxis, :]*STRIDE_CN;
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

##############
#  MAIN API  #
##############
class _sparse_matmul(torch.autograd.Function):
  
  sdd_cache = dict()
  dsd_cache = dict()
  dds_cache = dict()

  # Given an array sizes representing reduction size for each
  # column of a block-mode matrix multiplication,
  # performs load-balancing to achieve more smaller reductions
  # between `seg_size` elements
  @staticmethod
  def load_balance(sizes, block):
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
  
  ##########################
  # SPARSE = DENSE x DENSE #
  ##########################
  @staticmethod
  def make_sdd_lut(mask, block):
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
  def _sdd_matmul(a, b, trans_a, trans_b, trans_c,
                  mask, block, lut, locks, width, 
                  bench, time):
    if trans_c:
      a, b = b, a
      trans_a, trans_b = not trans_b, not trans_a
    AS0 = a.size(1 if trans_a else 0)
    AS1 = a.size(0 if trans_a else 1)
    BS0 = b.size(1 if trans_b else 0)
    BS1 = b.size(0 if trans_b else 1)
    dtype = a.dtype
    # create kernel
    key = (block, a.dtype, b.type, trans_a, trans_b, trans_c)
    if key not in _sparse_matmul.sdd_cache:
      defines =  {'TM': block, 'TN': block, 'TK': 16, 'TYPE': dtype,
                  'STRIDE_AM': '1' if trans_a else 'lda', 
                  'STRIDE_AK': 'lda' if trans_a else '1',
                  'STRIDE_BN': 'ldb' if trans_b else '1', 
                  'STRIDE_BK': '1' if trans_b else 'ldb',
                  'STRIDE_CM': 'ldc', 
                  'STRIDE_CN': '1',
                  'SDD': True, 'TZ': 2, 'NAME': 'sdd_kernel'}
      _sparse_matmul.sdd_cache[key] = triton.kernel(src, defines=defines)
    kernel = _sparse_matmul.sdd_cache[key]
    # create output
    c = torch.empty((width, block, block), dtype=dtype, device=a.device)
    time[0] = kernel(a, b, c, a.size(1), b.size(1), block, AS0, AS1, lut, locks, locks.size(1), 
                     grid = lambda opt: [opt.d('TZ'), width], 
                     bench = bench)
    # save for backward pass
    return c

  ##########################
  # DENSE = DENSE x SPARSE #
  ##########################
  
  # Given a binary mask of 0s and 1s,
  # Construct look-up table for efficient execution on GPUs
  @staticmethod
  def make_dxx_lut(mask, block, step, trans):
    # nonzeros
    if trans:
      sizes = torch.sum(mask, 1)
      nnz = torch.nonzero(mask)
    else:
      sizes = torch.sum(mask, 0)
      nnz = torch.nonzero(mask.T)
    num_blocks = nnz.size(0)
    # load-balancing
    segments, column, lockid, maxid, offsets = _sparse_matmul.load_balance(sizes, block)
    segments *= step
    # pointer increments
    offsets = torch.min(offsets, (num_blocks - 1)*torch.ones_like(offsets))
    idx = nnz[:, 1]*block
    xincs = idx.clone() 
    xincs[1:] -= idx[:-1]
    # divide block into multiple steps
    div = block // step
    xincs = xincs.view(-1, 1).repeat(1, div)
    xincs[:, 1:] = step
    xincs[:, 0 ] -= (div-1)*step
    # first increment for each reduction is actually the offset
    xincs[offsets[segments>0], 0] = idx[offsets[segments>0]]
    xincs = xincs.view(-1)
    # block-mode input increments
    if trans:
      widx = torch.arange(num_blocks)
    else:
      maskw = mask.clone()
      maskw[maskw > 0] = 1 + torch.arange(maskw.sum())
      widx = maskw.T[maskw.T > 0] - 1
    widx = widx * block * block
    wincs = widx.clone()
    wincs[1:] -= widx[:-1]
    wincs = wincs.view(-1, 1).repeat(1, div)
    if trans:
      wincs[:, 1:] = step
      wincs[:, 0] -= (div-1)*step
    else:
      wincs[:, 1:] = step*block
      wincs[:, 0] -= (div - 1)*step*block
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
  def _dds_matmul(a, b, trans_a, trans_b, trans_c,
              mask, block, lut, locks, width, 
              bench, time):
    # shapes / dtypes
    AS0 = a.size(1 if trans_a else 0)
    AS1 = a.size(0 if trans_a else 1)
    BS0 = block * mask.size(1 if trans_b else 0)
    BS1 = block * mask.size(0 if trans_b else 1)
    dtype = a.dtype
    # kernel
    key = (block, a.dtype, b.dtype, trans_a, trans_b, trans_c)
    if key not in _sparse_matmul.dds_cache:
      defines = {'TM': 128, 'TN': block, 'TK': 8, 
                 'TYPE': dtype,
                 'STRIDE_AM': 1 if trans_a else 'lda',
                 'STRIDE_AK': 'lda' if trans_a else 1,
                 'STRIDE_BN': block if trans_b else 1, 
                 'STRIDE_BK': 1 if trans_b else block,
                 'STRIDE_CM': '1' if trans_c else 'ldc',
                 'STRIDE_CN': 'ldc' if trans_c else '1',
                 'NAME': 'dds_kernel',
                 'DDS': True}
      _sparse_matmul.dds_cache[key] = triton.kernel(src, defines=defines)
    kernel = _sparse_matmul.dds_cache[key]
    # output
    CS0 = BS1 if trans_c else AS0
    CS1 = AS0 if trans_c else BS1
    c = torch.empty((CS0, CS1), dtype=dtype, device=a.device)
    time[0] = kernel(a, b, c, a.size(1), BS1, c.size(1), AS0, AS1, lut, locks, locks.size(1), 
                     grid = lambda opt: [width, triton.cdiv(AS0, opt.d('TM'))], 
                     bench = bench)
    return c
  
  @staticmethod
  def _dsd_matmul(a, b, trans_a, trans_b, trans_c,
                  mask, block, lut, locks, width,
                  bench, time):
    # shapes / dtypes
    AS0 = block * mask.size(1 if trans_a else 0)
    AS1 = block * mask.size(0 if trans_a else 1)
    BS0 = b.size(1 if trans_b else 0)
    BS1 = b.size(0 if trans_b else 1)
    dtype = a.dtype
    # kernel
    key = (block, a.dtype, b.dtype, trans_a, trans_b, trans_c)
    if key not in _sparse_matmul.dsd_cache:
      defines = {'TM': block, 'TN': 128, 'TK': 8, 
                 'TYPE': dtype,
                 'STRIDE_AM': 1 if trans_a else block, 
                 'STRIDE_AK': block if trans_a else 1,
                 'STRIDE_BN': 'ldb' if trans_b else '1',
                 'STRIDE_BK': '1' if trans_b else 'ldb',
                 'STRIDE_CM': '1' if trans_c else 'ldc',
                 'STRIDE_CN': 'ldc' if trans_c else '1',
                 'NAME': 'dsd_kernel',
                 'DSD': True}
      _sparse_matmul.dsd_cache[key] = triton.kernel(src, defines=defines)
    kernel = _sparse_matmul.dsd_cache[key]
    # output
    CS0 = BS1 if trans_c else AS0
    CS1 = AS0 if trans_c else BS1
    c = torch.empty((CS0, CS1), dtype=dtype, device=a.device)
    time[0] = kernel(a, b, c, AS1, b.size(1), c.size(1), AS0, AS1, lut, locks, locks.size(1), 
                     grid = lambda opt: [width, triton.cdiv(BS1, opt.d('TN'))], 
                     bench = bench)
    return c

  fn = {'sdd': _sdd_matmul.__get__(object),
        'dsd': _dsd_matmul.__get__(object),
        'dds': _dds_matmul.__get__(object)}

  @staticmethod
  def forward(ctx, a, b, trans_a, trans_b, trans_c,
              mode, mask, block,
              c_lut, c_locks, c_width, c_bench, c_time,
              da_lut, da_locks, da_width, da_bench, da_time,
              db_lut, db_locks, db_width, db_bench, db_time):
    c = _sparse_matmul.fn[mode](a, b, trans_a, trans_b, trans_c, mask, block, 
                                c_lut, c_locks, c_width, c_bench, c_time)
    # save for backward
    ctx.save_for_backward(a, da_lut, da_locks, b, db_lut, db_locks)
    ctx.da_width = da_width
    ctx.da_bench = da_bench
    ctx.da_time = da_time
    ctx.db_width = db_width
    ctx.db_bench = db_bench
    ctx.db_time = db_time
    ctx.mode = mode
    ctx.mask = mask
    ctx.block = block
    ctx.trans_a = trans_a
    ctx.trans_b = trans_b
    return c

  @staticmethod
  def backward(ctx, dc):
    # saved for backward
    a, da_lut, da_locks, b, db_lut, db_locks = ctx.saved_tensors
    mode = ctx.mode
    # gradients w.r.t. a
    if ctx.needs_input_grad[0]:
      mode_da = mode[1] + mode[0] + mode[2]
      da = _sparse_matmul.fn[mode_da](dc, b, False, not ctx.trans_b, ctx.trans_a, ctx.mask, ctx.block,
                         da_lut, da_locks, ctx.da_width, ctx.da_bench, ctx.da_time)
    # gradients w.r.t. b
    if ctx.needs_input_grad[1]:
      mode_db = mode[2] + mode[1] + mode[0]
      db = _sparse_matmul.fn[mode_db](a, dc, not ctx.trans_a, False, ctx.trans_b, ctx.mask, ctx.block,
                         db_lut, db_locks, ctx.db_width, ctx.db_bench, ctx.db_time)
    return da, db, None, None, None,\
           None, None, None,\
           None, None, None, None, None,\
           None, None, None, None, None,\
           None, None, None, None, None

class SparseMatMul:

  def __init__(self, trans_a, trans_b, mode, mask, block, bench = False):
    # C look-up table
    if mode == 'sdd':
      self.c_lut, self.c_locks, self.c_width = _sparse_matmul.make_sdd_lut(mask, block)
    elif mode == 'dsd':
      self.c_lut, self.c_locks, self.c_width = _sparse_matmul.make_dxx_lut(mask, block, 8, not trans_a)
    elif mode == 'dds':
      self.c_lut, self.c_locks, self.c_width = _sparse_matmul.make_dxx_lut(mask, block, 8, trans_b)
    # DA look-up table
    if mode == 'sdd':
      self.da_lut, self.da_locks, self.da_width = _sparse_matmul.make_dxx_lut(mask, block, 8, True)
    elif mode == 'dsd':
      self.da_lut, self.da_locks, self.da_width = _sparse_matmul.make_sdd_lut(mask, block)
    elif mode == 'dds':
      self.da_lut, self.da_locks, self.da_width = _sparse_matmul.make_dxx_lut(mask, block, 8, not trans_b)
    # DB look-up table
    if mode == 'sdd':
      self.db_lut, self.db_locks, self.db_width = _sparse_matmul.make_dxx_lut(mask, block, 8, False)
    elif mode == 'dsd':
      self.db_lut, self.db_locks, self.db_width = _sparse_matmul.make_dxx_lut(mask, block, 8, trans_a)
    elif mode == 'dds':
      self.db_lut, self.db_locks, self.db_width = _sparse_matmul.make_sdd_lut(mask, block)
    # attributes
    self.trans_a = trans_a
    self.trans_b = trans_b
    self.mode = mode
    self.mask = mask
    self.block = block
    # timings
    self.bench = bench
    self.time_c = None
    self.time_da = None
    self.time_db = None
  
  def __call__(self, a, b):
    time_c  = [None]
    time_da = [None]
    time_db = [None]
    c = _sparse_matmul.apply(a, b, self.trans_a, self.trans_b, False,
                              self.mode, self.mask, self.block,
                              self.c_lut, self.c_locks, self.c_width, self.bench, time_c,
                              self.da_lut, self.da_locks, self.da_width, self.bench, time_da,
                              self.db_lut, self.db_locks, self.db_width, self.bench, time_db)
    self.time_c = time_c[0]
    self.time_da = time_da[0]
    self.time_db = time_db[0]
    return c


class Linear(torch.nn.Module):

  def __init__(self, in_features, out_features, block, mask, bench = False):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.block = block
    self.mask = mask
    self.weight = torch.nn.Parameter(torch.Tensor(mask.sum(), block, block))
    self.reset_parameters()
    self.matmul = SparseMatMul(False, False, 'dds', mask, block, bench)
  
  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
  def forward(self, input):
    return self.matmul(input, self.weight)