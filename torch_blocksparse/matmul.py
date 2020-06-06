import triton
import torch
import math

src = '''
  __global__ void NAME (TYPE* A __readonly  __noalias __aligned(16),
                        TYPE* B __readonly  __noalias __aligned(16),
                        TYPE* C __noalias __aligned(16),
                        int lda __multipleof(8), 
                        int ldb __multipleof(8), 
                        int ldc __multipleof(8),
                        long stride_za __multipleof(8),
                        long stride_zb __multipleof(8),
                        long stride_zc __multipleof(8),
                        long stride_ha __multipleof(8),
                        long stride_hb __multipleof(8),
                        long stride_hc __multipleof(8),
                        int DS0, int DS1,
                        int SDD_K __multipleof(16), 
                        int SDD_off_width,
                        int* lut, int* locks, int nlocks) {
    /* ---------------- */
    /*    Prologue      */
    /* ---------------- */
    // program ids
    int pid0 = get_program_id(0);
    int pid1 = get_program_id(1);
    int pidz = get_program_id(2);
#ifdef SDD
    // load LUT header
    pid1 = pid1 + SDD_off_width;
    int blockidm[TM] = (0 ... TM) / BLOCK;
    int blockidn[TN] = (0 ... TN) / BLOCK;
    int offlutm[TM]  = blockidm*(TN/BLOCK)*4;
    int offlutn[TN]  = blockidn*4;
    int *header      = lut + pid1 * (TM/BLOCK) * (TN/BLOCK) * 4;
    int z            = *(header + 0);
    int i[TM]        = *(header + 1 + offlutm);
    int j[TN]        = *(header + 2 + offlutn);
    int AS1 = SDD_K / TZ;
    int lockid = select(TZ > 1, 1, 0);
    int offka  = pid0 * AS1;
    int offkb  = pid0 * AS1;
    int offmc  = 0;
    int offnc  = 0;
    int offpa  = 0;
    int offpb  = 0;
    int maxid = TZ;
    int offhc = 0;
    int offha = z;
    int offhb = z;
    int ram[TM] = i*BLOCK + ((0 ... TM) % BLOCK);
    int rbn[TN] = j*BLOCK + ((0 ... TN) % BLOCK);
#else
    // load LUT header
    int *header = lut + pid0 * 6;
    int offset = *(header + 0);
    int AS1    = *(header + 1);
    int column = *(header + 2);
    int depth  = *(header + 3);
    int lockid = *(header + 4);
    int maxid  = *(header + 5);
    int *pinc  = lut + offset;
    int offhc = depth;
#ifdef DSD
    // output offset
    int offnc = pid1 * TN;
    int offmc = column * TM;
    int offpc = 0;
    // dense input offset
    int offnb = pid1 * TN;
    int offkb __multipleof(8) = *pinc;
    int offpb = 0;
    // sparse input offset
    int offma = 0;
    int offka = 0;
    long offpa __multipleof(8) = *(pinc + 1);
    offpa = offpa * BLOCK * BLOCK;
    int offha = 0;
    int offhb = depth;
#endif
#ifdef DDS
    // output offset
    int offmc = pid1 * TM;
    int offnc = column * TN;
    int offpc = 0;
    // dense input offset
    int offma = pid1 * TM;
    int offka __multipleof(8) = *pinc;
    int offpa = 0;
    // sparse input offset
    int offnb = 0;
    int offkb = 0;
    long offpb __multipleof(8) = *(pinc + 1);
    offpb = offpb * BLOCK * BLOCK;
    int offha = depth;
    int offhb = 0;
#endif
    int ram[TM] = offma + 0 ... TM;
    int rbn[TN] = offnb + 0 ... TN;
#endif
    // initialize a, b pointers
    int rka[TK] = offka + 0 ... TK;
    int rkb[TK] = offkb + 0 ... TK;
    TYPE* pa[TM, TK] = A + pidz * stride_za + offha * stride_ha + offpa + ram[:, newaxis] * STRIDE_AM + rka[newaxis, :] * STRIDE_AK;
    TYPE* pb[TK, TN] = B + pidz * stride_zb + offhb * stride_hb + offpb + rbn[newaxis, :] * STRIDE_BN + rkb[:, newaxis] * STRIDE_BK;
    // pre-fetch
#ifdef DDS
    bool checkam[TM, TK] = ram[:, newaxis] < DS0;
#else
    bool checkam[TM, TK] = AS1 > 0;
#endif
#ifdef DSD
    bool checkbn[TK, TN] = rbn[newaxis, :] < DS0;
#else
    bool checkbn[TK, TN] = AS1 > 0;
#endif
    TYPE a[TM, TK] = checkam ? *pa : 0;
    TYPE b[TK, TN] = checkbn ? *pb : 0;

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
      bool checkak[TM, TK] = k > TK;
      bool checkbk[TK, TN] = k > TK;
      bool checka[TM, TK] = checkam && checkak;
      bool checkb[TK, TN] = checkbk && checkbn;
      a = *?(checka)pa;
      b = *?(checkb)pb;
    }
    TYPE c[TM, TN] = acc;

    /* ---------------- */
    /*    Epilogue      */
    /* ---------------- */
    // initialize c pointers
#ifdef SDD
    bool checkc[TM, TN] = 1;
    // rematerialize
    int rr_blockidm[TM]  = (0 ... TM) / BLOCK;
    int rr_blockidn[TN]  = (0 ... TN) / BLOCK;
    int rr_offlutm[TM]   = rr_blockidm*(TN/BLOCK)*4;
    int rr_offlutn[TN]   = rr_blockidn*4;
    int off_bkid[TM, TN] = 3 + rr_offlutm[:, newaxis] + rr_offlutn[newaxis, :];
    int bkid[TM, TN]     = *(header + off_bkid);
    long offpc[TM, TN]   = bkid * BLOCK * BLOCK;
    // range within blocks
    int   rcm[TM]    = (0 ... TM) % BLOCK;
    int   rcn[TN]    = (0 ... TN) % BLOCK;
#else
    int   rcm[TM]    = offmc + 0 ... TM;
    int   rcn[TN]    = offnc + 0 ... TN;
#ifdef DSD
    bool checkc[TM, TN] = rcn[newaxis, :] < DS0;
#endif
#ifdef DDS
    bool checkc[TM, TN] = rcm[:, newaxis] < DS0;
#endif
#endif
    TYPE* pc[TM, TN] = C + offpc + offhc*stride_hc + pidz*stride_zc + rcm[:, newaxis]*STRIDE_CM + rcn[newaxis, :]*STRIDE_CN;
    // write-back directly
    if(lockid == 0) {
      *?(checkc) pc = c;
    }
    // accumulate partial result using spin-locks
    else {
      int *plock = locks + get_program_id(2)*nlocks*get_num_programs(1) + get_program_id(1)*nlocks + lockid - 1;
      int *pcount = plock + get_num_programs(2)*get_num_programs(1)*nlocks;
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
  locks = None

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
    min_size = sizes[sizes != 0].min()
    #if max_size > min_size * 2.0:
    #  seg_max = max(triton.cdiv(max_size, 4), min_size*2)
    #else:
    #  seg_max = max_size
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
  
  @staticmethod
  def get_locks(size):
    if _sparse_matmul.locks is None or \
        size > _sparse_matmul.locks.size(0):
      _sparse_matmul.locks = torch.zeros(size, dtype=torch.int32).cuda()
    return _sparse_matmul.locks

  ##########################
  # SPARSE = DENSE x DENSE #
  ##########################
  _sdd_segment_src = '''

#ifdef _OPENMP
#include <omp.h>
#endif

typedef std::vector<std::tuple<int, torch::Tensor>> ret_t;

void segment_blocks(torch::Tensor layout, torch::Tensor idx, torch::Tensor scratch, int max_width, ret_t& ret){
  size_t H = layout.size(0);
  size_t M = layout.size(1);
  size_t N = layout.size(2);
  torch::Tensor tmp = torch::zeros_like(layout);

  
  auto _tmp     = tmp.accessor    <int, 3>();
  auto _layout  = layout.accessor <int, 3>();
  auto _idx     = idx.accessor    <int, 3>();
  auto _scratch = scratch.accessor<int, 3>();
  std::vector<int> current(H, 0);


  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(size_t h = 0; h < H; h++){

    // surrounding indices
    std::vector<int>              ii_left(max_width, -1);
    std::vector<std::vector<int>> ii_top(max_width, std::vector<int>(N, -1));

    for(size_t m = 0; m < M; m++){
      for(size_t n = 0; n < N; n++){
        int v     = _layout[h][m][n];
        if(v == 0)
          continue;
        int n_left= ii_left[max_width-1];
        int m_top = ii_top [max_width-1][n];
        int top      = (m_top >= 0)               ? _tmp[h][m_top][n]      : 0;
        int left     = (n_left >= 0)              ? _tmp[h][m][n_left]     : 0;
        int topleft  = (m_top >=0 && n_left >= 0) ? _tmp[h][m_top][n_left] : 0;
        int width    = std::min(left, std::min(top, topleft)) + 1;

        // reset width if blocks cannot be 
        // packed together (i.e., there's a 1 "in the middle")
        for(int nn = n_left + 1; nn < n; nn++)
          if(ii_top[max_width-1][nn] > ii_top[max_width-1][n])
            width = 1;
        _tmp[h][m][n] = width;

        // update n_left ring buffer
        for(int k = 0; k < max_width-1; k++)
          ii_left[k] = ii_left[k+1];
        ii_left[max_width-1] = n;

        // update ii_top ring buffer
        for(int k = 0; k < max_width-1; k++)
          ii_top[k][n] = ii_top[k+1][n];
        ii_top[max_width-1][n] = m;

        // block is too small -- skip
        if(width != max_width)
          continue;

        // retained blocks are set to zeros
        for(size_t km = 0; km < max_width; km++)
        for(size_t kn = 0; kn < max_width; kn++)
        {
          int mm = ii_top[km][n];
          int nn = ii_left[kn];
          if(mm < 0 || nn < 0)
            continue;
          _layout[h][mm][nn] = 0;
          _tmp[h][mm][nn] = 0;
          _scratch[h][current[h]][0] = (int)h;
          _scratch[h][current[h]][1] = (int)mm;
          _scratch[h][current[h]][2] = (int)nn;
          _scratch[h][current[h]][3] = _idx[h][mm][nn];
          current[h]++;
        }
      }
    }
  }
  std::vector<torch::Tensor> to_cat;
  for(size_t h = 0; h < H; h++)
    if(current[h] > 0)
      to_cat.push_back(scratch[h].slice(0, 0, current[h]));
  if(!to_cat.empty())
    ret.push_back({max_width, torch::cat(to_cat)});
}

ret_t sdd_segment(torch::Tensor layout, int start_width) {
  ret_t ret;

  // block index
  torch::Tensor idx = torch::zeros_like(layout);
  int current = 0;
  size_t H = layout.size(0);
  size_t M = layout.size(1);
  size_t N = layout.size(2);
  auto _layout  = layout.accessor <int, 3>();
  auto _idx = idx.accessor<int, 3>();
  for(size_t h = 0; h < H; h++)
  for(size_t m = 0; m < M; m++)
  for(size_t n = 0; n < N; n++){
    if(_layout[h][m][n] == 0)
      continue;
    _idx[h][m][n] = current++;
  }

  // scratch memory
  torch::Tensor scratch = torch::empty({H, layout.sum().item<int>(), 4}, layout.dtype());

  for(int max_width = start_width; max_width > 0; max_width /= 2)
    segment_blocks(layout, idx, scratch, max_width, ret);
  return ret;
}
'''

  sdd_segment = torch.utils.cpp_extension.load_inline(name='sdd_segment', 
                                                      cpp_sources=[_sdd_segment_src], 
                                                      functions=['sdd_segment'],
                                                      extra_cflags=['-O2', '-fopenmp']).sdd_segment

  @staticmethod
  def make_sdd_lut(layout, block, dtype):
    start_width = 64 // block
    segmented = _sparse_matmul.sdd_segment(layout.type(torch.int32), start_width)
    luts, widths, packs = [], [], []
    for size, nnz in segmented:
      width = nnz.shape[0] // (size*size)
      h = nnz[:, 0]
      i = nnz[:, 1]
      j = nnz[:, 2]
      b = nnz[:, 3]
      lut = torch.stack((h, i, j, b), dim=1).view(-1).contiguous()
      luts.append(lut.type(torch.int32).cuda()) 
      widths.append(width)
      packs.append(size)
    # create locks
    return luts, None, widths, packs

  @staticmethod
  def _sdd_matmul(a, b, trans_a, trans_b, trans_c,
                  spdims, block, luts, num_locks, widths, packs,
                  bench, time):
    if trans_c:
      a, b = b, a
      trans_a, trans_b = not trans_b, not trans_a
    AS0 = a.size(0)
    AS1 = a.size(1)
    AS2 = a.size(3 if trans_a else 2)
    AS3 = a.size(2 if trans_a else 3)
    BS0 = b.size(0)
    BS1 = b.size(1)
    BS2 = b.size(3 if trans_b else 2)
    BS3 = b.size(2 if trans_b else 3)
    dtype = a.dtype
    # create kernel
    total_width = sum([width*pack*pack for width,pack in zip(widths, packs)])
    c = torch.empty((AS0, total_width, block, block), dtype=dtype, device=a.device)
    for lut, width, pack in zip(luts, widths, packs):
      num_lock = 1
      key = (block, a.dtype, b.dtype, trans_a, trans_b, trans_c, pack)
      if key not in _sparse_matmul.sdd_cache:
        TK = {torch.float32: [8, 16],
              torch.float16: [16, 32, 64]}[dtype]
        defines =  {'TM': block*pack, 'TN': block*pack, 'TMN': block*block*pack*pack, 'BLOCK': block, 
                  'TK': TK, 'TYPE': dtype,
                  'STRIDE_AM': '1'    if trans_a else 'lda', 
                  'STRIDE_AK': 'lda'  if trans_a else '1',
                  'STRIDE_BN': 'ldb'  if trans_b else '1', 
                  'STRIDE_BK': '1'    if trans_b else 'ldb',
                  'STRIDE_CM': 'ldc', 
                  'STRIDE_CN': '1',
                  'SDD': True, 'TZ': 1, 'NAME': 'sdd_kernel'}
        _sparse_matmul.sdd_cache[key] = triton.kernel(src, defines=defines, num_warps=[1, 2, 4])

      kernel = _sparse_matmul.sdd_cache[key]
      # create output
      locks = _sparse_matmul.get_locks(2*width*AS0*num_lock)
      # maximum grid size is 65535
      # so operation might be decomposed into multiple
      # kernel calls
      max_width = 49152
      total = 0 if bench else None
      for off_width in range(0, width, max_width):
        current  = kernel(a, b, c, 
                          a.stride(2), b.stride(2), block, 
                          a.stride(0), b.stride(0), c.stride(0),
                          a.stride(1), b.stride(1), c.stride(0), 
                          AS2, AS2, AS3, off_width, lut, locks, num_lock, 
                          grid = lambda opt: [opt.d('TZ'), min(max_width, width - off_width), AS0], 
                          bench = bench)
        total = total + current if bench else None
      time[0] = total
    # save for backward pass
    return c

  ##########################
  # DENSE = DENSE x SPARSE #
  ##########################
  
  # Given a binary layout of 0s and 1s,
  # Construct look-up table for efficient execution on GPUs
  @staticmethod
  def make_dxx_lut(layout, block, step, trans, transform = lambda idx: idx):
    # load-balancing
    _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
    segments = _empty.clone()
    column   = _empty.clone()
    depth    = _empty.clone()
    lockid   = _empty.clone()
    maxid    = _empty.clone()
    offsets  = _empty.clone()
    current_offset = 0
    current_maxid = 0
    for z in range(layout.size(0)):
      if trans:
        sizes = torch.sum(layout[z, :, :], 1)
      else:
        sizes = torch.sum(layout[z, :, :], 0)
      z_segments, z_column, z_lockid, z_maxid, z_offsets = _sparse_matmul.load_balance(sizes, block)
      z_depth = z * torch.ones_like(z_segments)
      z_lockid[z_lockid > 0] += current_maxid
      current_maxid = z_lockid.max()
      # concatenate depth
      segments = torch.cat((segments, z_segments))
      column   = torch.cat((column,    z_column))
      depth    = torch.cat((depth,     z_depth))
      maxid    = torch.cat((maxid,     z_maxid))
      offsets  = torch.cat((offsets,   current_offset + z_offsets))
      lockid   = torch.cat((lockid,    z_lockid))
      current_offset += layout[z, :, :].sum()
    segments *= step
    # pointer increments
    if trans:
      nnz = layout.nonzero()
    else:
      nnz = layout.transpose(1, 2).nonzero()
    num_blocks = nnz.size(0)
    offsets = torch.min(offsets, (num_blocks - 1)*torch.ones_like(offsets))
    idx = transform(nnz[:, 2]*block)
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
      widx = _empty.clone()
      current_offset = 0
      for z in range(layout.size(0)):
        layoutw = layout[z, :, :].clone()
        msum = layoutw.sum()
        layoutw[layoutw > 0] = 1 + torch.arange(msum)
        widx = torch.cat((widx, current_offset + layoutw.T[layoutw.T > 0] - 1))
        current_offset += msum
    widx = widx
    wincs = widx*block*block
    wincs[1:] -= widx[:-1]*block*block
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
    offsets += 6*width
    header = torch.stack((offsets, segments, column, depth, lockid, maxid), dim=1).view(-1).contiguous()
    incs = torch.stack((xincs, wincs), dim=1).view(-1).contiguous()
    # create lut
    lut = torch.cat((header, incs))
    lut = lut.type(torch.int32).cuda()
    # create locks
    num_locks = max(1, lockid.max())
    return lut, num_locks, width, None

  @staticmethod
  def _dds_matmul(a, b, trans_a, trans_b, trans_c,
              spdims, block, lut, num_locks, width, packs,
              bench, time):
    # shapes / dtypes
    AS0 = a.size(0)
    AS1 = a.size(1)
    AS2 = a.size(3 if trans_a else 2)
    AS3 = a.size(2 if trans_a else 3)
    BS0 = spdims[0]
    BS1 = block * spdims[2 if trans_b else 1]
    BS2 = block * spdims[1 if trans_b else 2]
    dtype = a.dtype
    # kernel
    key = (block, a.dtype, b.dtype, trans_a, trans_b, trans_c)
    if key not in _sparse_matmul.dds_cache:
      TM = [64, 128] if dtype == torch.float32 else [64, 128, 256]
      TK = [8]       if dtype == torch.float32 else [16]
      defines = {'TM': TM, 'TN': block, 'TK': TK, 
                 'BLOCK': block,
                 'TYPE': dtype,
                 'STRIDE_AM': 1 if trans_a else 'lda',
                 'STRIDE_AK': 'lda' if trans_a else 1,
                 'STRIDE_BN': block if trans_b else 1, 
                 'STRIDE_BK': 1 if trans_b else block,
                 'STRIDE_CM': '1' if trans_c else 'ldc',
                 'STRIDE_CN': 'ldc' if trans_c else '1',
                 'NAME': 'dds_kernel',
                 'DDS': True}
      _sparse_matmul.dds_cache[key] = triton.kernel(src, defines=defines, num_warps=[4])
    kernel = _sparse_matmul.dds_cache[key]
    # output
    CS0 = AS0
    CS1 = AS1
    CS2 = BS2 if trans_c else AS2
    CS3 = AS2 if trans_c else BS2
    locks = _sparse_matmul.get_locks(2*AS0*AS2//32*num_locks)
    c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
    time[0] = kernel(a, b, c, 
                     a.stride(2), block, c.stride(2), 
                     a.stride(0), b.stride(0), c.stride(0),
                     a.stride(1), b.stride(1), c.stride(1),
                     AS2, BS2, 0, 0, lut, locks, num_locks, 
                     grid = lambda opt: [width, triton.cdiv(AS2, opt.d('TM')), AS0], 
                     bench = bench)
    return c
  
  @staticmethod
  def _dsd_matmul(a, b, trans_a, trans_b, trans_c,
                  spdims, block, lut, num_locks, width, packs,
                  bench, time):
    # shapes / dtypes
    AS0 = spdims[0]
    AS1 = block * spdims[2 if trans_a else 1]
    AS2 = block * spdims[1 if trans_a else 2]
    BS0 = b.size(0)
    BS1 = b.size(1)
    BS2 = b.size(3 if trans_b else 2)
    BS3 = b.size(2 if trans_b else 3)
    dtype = a.dtype
    # kernel
    key = (block, a.dtype, b.dtype, trans_a, trans_b, trans_c)
    if key not in _sparse_matmul.dsd_cache:
      TN = [64, 128] if dtype == torch.float32 else [64, 128, 256]
      TK = [8]       if dtype == torch.float32 else [16]
      defines = {'TM': block, 'TN': TN, 'TK': TK, 
                 'BLOCK': block,
                 'TYPE': dtype,
                 'STRIDE_AM': 1 if trans_a else block, 
                 'STRIDE_AK': block if trans_a else 1,
                 'STRIDE_BN': 'ldb' if trans_b else '1',
                 'STRIDE_BK': '1' if trans_b else 'ldb',
                 'STRIDE_CM': '1' if trans_c else 'ldc',
                 'STRIDE_CN': 'ldc' if trans_c else '1',
                 'NAME': 'dsd_kernel',
                 'DSD': True}
      _sparse_matmul.dsd_cache[key] = triton.kernel(src, defines=defines, num_warps=[4])
    kernel = _sparse_matmul.dsd_cache[key]
    # output
    CS0 = BS0
    CS1 = BS1
    CS2 = BS3 if trans_c else AS1
    CS3 = AS1 if trans_c else BS3
    locks = _sparse_matmul.get_locks(2*BS0*BS3//32*num_locks)
    c = torch.empty((CS0, CS1, CS2, CS3), dtype=dtype, device=a.device)
    time[0] = kernel(a, b, c, 
                     block, b.stride(2), c.stride(2), 
                     a.stride(0), b.stride(0), c.stride(0),
                     a.stride(1), b.stride(1), c.stride(1),
                     BS3, AS1, 0, 0, lut, locks, num_locks, 
                     grid = lambda opt: [width, triton.cdiv(BS3, opt.d('TN')), BS0], 
                     bench = bench)
    return c

  fn = {'sdd': _sdd_matmul.__get__(object),
        'dsd': _dsd_matmul.__get__(object),
        'dds': _dds_matmul.__get__(object)}

  @staticmethod
  def forward(ctx, a, b, trans_a, trans_b, trans_c,
              mode, spdims, block,
              c_lut, c_num_locks, c_width, c_packs, c_bench, c_time,
              da_lut, da_num_locks, da_width, da_packs, da_bench, da_time,
              db_lut, db_num_locks, db_width, db_packs, db_bench, db_time):
    c = _sparse_matmul.fn[mode](a, b, trans_a, trans_b, trans_c, spdims, block, 
                                c_lut, c_num_locks, c_width, c_packs, c_bench, c_time)
    # save for backward
    ctx.save_for_backward(a, b)
    ctx.da_num_locks = da_num_locks
    ctx.da_lut   = da_lut
    ctx.da_width = da_width
    ctx.da_packs = da_packs
    ctx.da_bench = da_bench
    ctx.da_time = da_time
    ctx.db_lut = db_lut
    ctx.db_num_locks = db_num_locks
    ctx.db_width = db_width
    ctx.db_bench = db_bench
    ctx.db_packs = db_packs
    ctx.db_time = db_time
    ctx.mode = mode
    ctx.spdims = spdims
    ctx.block = block
    ctx.trans_a = trans_a
    ctx.trans_b = trans_b
    return c

  @staticmethod
  def backward(ctx, dc):
    # saved for backward
    a, b = ctx.saved_tensors
    mode = ctx.mode
    # gradients w.r.t. a
    if ctx.needs_input_grad[0]:
      mode_da = mode[1] + mode[0] + mode[2]
      da = _sparse_matmul.fn[mode_da](dc, b, False, not ctx.trans_b, ctx.trans_a, ctx.spdims, ctx.block,
                         ctx.da_lut, ctx.da_num_locks, ctx.da_width, ctx.da_packs, ctx.da_bench, ctx.da_time)
    # gradients w.r.t. b
    if ctx.needs_input_grad[1]:
      mode_db = mode[2] + mode[1] + mode[0]
      db = _sparse_matmul.fn[mode_db](a, dc, not ctx.trans_a, False, ctx.trans_b, ctx.spdims, ctx.block,
                         ctx.db_lut, ctx.db_num_locks, ctx.db_width, ctx.db_packs, ctx.db_bench, ctx.db_time)
    return da, db, None, None, None,\
           None, None, None, None,\
           None, None, None, None, None, None,\
           None, None, None, None, None, None,\
           None, None, None, None, None, None

class MatMul:
  
  
  def make_lut(self, dtype):
    key = (dtype, )
    if key in self.lut_cache:
      return self.lut_cache[key]
    # C look-up table
    layout, block = self.layout, self.block
    step = 8 if dtype == torch.float32 else 16
    if self.mode == 'sdd':
      c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_sdd_lut(layout, block, dtype)
    elif self.mode == 'dsd':
      c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_dxx_lut(layout, block, step, not self.trans_a)
    elif self.mode == 'dds':
      c_lut, c_num_locks, c_width, c_packs = _sparse_matmul.make_dxx_lut(layout, block, step, self.trans_b)
    # DA look-up table
    if self.mode == 'sdd':
      da_lut, da_num_locks, da_width, da_packs = _sparse_matmul.make_dxx_lut(layout, block, step, True)
    elif self.mode == 'dsd':
      da_lut, da_num_locks, da_width, da_packs = _sparse_matmul.make_sdd_lut(layout, block, dtype)
    elif self.mode == 'dds':
      da_lut, da_num_locks, da_width, da_packs = _sparse_matmul.make_dxx_lut(layout, block, step, not self.trans_b)
    # DB look-up table
    if self.mode == 'sdd':
      db_lut, db_num_locks, db_width, db_packs = _sparse_matmul.make_dxx_lut(layout, block, step, False)
    elif self.mode == 'dsd':
      db_lut, db_num_locks, db_width, db_packs = _sparse_matmul.make_dxx_lut(layout, block, step, self.trans_a)
    elif self.mode == 'dds':
      db_lut, db_num_locks, db_width, db_packs = _sparse_matmul.make_sdd_lut(layout, block, dtype)
    self.lut_cache[key] = (c_lut, c_num_locks, c_width, c_packs,\
                           da_lut, da_num_locks, da_width, da_packs,\
                           db_lut, db_num_locks, db_width, db_packs)
    return self.lut_cache[key]

  def __init__(self, layout, block, mode, trans_a = False, trans_b = False, bench = False):
    if mode not in ['sdd', 'dsd', 'dds']:
      raise NotImplementedError('Supported modes are: sdd, dsd, dds')
    # look-up table cache
    self.lut_cache = dict()
    # attributes
    self.trans_a = trans_a
    self.trans_b = trans_b
    self.mode = mode
    self.spdims = layout.shape
    self.block = block
    self.layout = layout
    # timings
    self.bench = bench
    self.time_c = None
    self.time_da = None
    self.time_db = None
  
  # pad shapes of a tensor to make it
  # compatible with kernel calls
  @staticmethod
  def _pad_shape(x, is_sparse):
    max_dim = 3 if is_sparse else 4
    for i in range(max_dim - x.dim()):
      x = x.unsqueeze(0)
    return x

  def __call__(self, a, b):
    c_lut, c_num_locks, c_width, c_packs,\
    da_lut, da_num_locks, da_width, da_packs,\
    db_lut, db_num_locks, db_width, db_packs = self.make_lut(a.dtype)
    # timings
    time_c  = [None]
    time_da = [None]
    time_db = [None]
    # pad shapes with ones
    a = MatMul._pad_shape(a, self.mode == 'dsd')
    b = MatMul._pad_shape(b, self.mode == 'dds')
    # execute
    c = _sparse_matmul.apply(a, b, self.trans_a, self.trans_b, False,
                              self.mode, self.spdims, self.block,
                              c_lut, c_num_locks, c_width,    c_packs,  self.bench, time_c,
                              da_lut, da_num_locks, da_width, da_packs, self.bench, time_da,
                              db_lut, db_num_locks, db_width, db_packs, self.bench, time_db)
    self.time_c = time_c[0]
    self.time_da = time_da[0]
    self.time_db = time_db[0]
    return c


class Linear(torch.nn.Module):

  def __init__(self, in_features, out_features, block, layout, bench = False):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.block = block
    self.weight = torch.nn.Parameter(torch.Tensor(layout.sum(), block, block))
    self.reset_parameters()
    self.matmul = MatMul(False, False, 'dds', layout, block, bench)
  
  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
  def forward(self, input):
    return self.matmul(input, self.weight)