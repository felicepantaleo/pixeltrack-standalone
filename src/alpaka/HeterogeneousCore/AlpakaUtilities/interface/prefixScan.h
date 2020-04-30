#ifndef HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h
#define HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h

#include <cstdint>
#include <HeterogeneousCore/AlpakaCore/alpakaConfig.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

template <typename T>
ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T const* __restrict__ ci, T* __restrict__ co, uint32_t i, uint32_t mask) {
  // ci and co may be the same
  auto x = ci[i];
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
ALPAKA_FN_INLINE void warpPrefixScan(uint32_t laneId, T* c, uint32_t i, uint32_t mask) {
  auto x = c[i];
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(mask, x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
  namespace Alpaka {
    // limited to 32*32 elements....
    struct blockPrefixScan {
    template <typename T, typename T_Acc>
    ALPAKA_FN_INLINE  void operator()(const T_Acc& acc, T const* __restrict__ ci,
                                                             T* __restrict__ co,
                                                             uint32_t size,
                                                             T* ws
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
                                                             = nullptr
#endif
    ) const {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

      uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDimension % 32);
      auto first = blockThreadIdx;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i += blockDimension) {
        auto laneId = blockThreadIdx & 0x1f;
        warpPrefixScan(laneId, ci, co, i, mask);
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = co[i];
        mask = __ballot_sync(mask, i + blockDimension < size);
      }
      alpaka::block::sync::syncBlockThreads(acc);
      if (size <= 32)
        return;
      if (blockThreadIdx < 32)
        auto laneId = blockThreadIdx & 0x1f;
        warpPrefixScan(laneId, ws, blockThreadIdx, 0xffffffff);
      alpaka::block::sync::syncBlockThreads(acc);
      for (auto i = first + 32; i < size; i += blockDimension) {
        auto warpId = i / 32;
        co[i] += ws[warpId - 1];
      }
      alpaka::block::sync::syncBlockThreads(acc);
#else
      co[0] = ci[0];
      for (uint32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
#endif
    }

    template <typename T, typename T_Acc>
    ALPAKA_FN_INLINE void operator()(const T_Acc& acc, 
                                                            T* c,
                                                            uint32_t size,
                                                            T* ws
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
                                                            = nullptr
#endif
    ) const {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      assert(ws);
      assert(size <= 1024);
      assert(0 == blockDimension % 32);
      auto first = blockThreadIdx;
      auto mask = __ballot_sync(0xffffffff, first < size);

      for (auto i = first; i < size; i += blockDimension) {
        warpPrefixScan(c, i, mask);
        auto laneId = blockThreadIdx & 0x1f;
        auto warpId = i / 32;
        assert(warpId < 32);
        if (31 == laneId)
          ws[warpId] = c[i];
        mask = __ballot_sync(mask, i + blockDimension < size);
      }
      alpaka::block::sync::syncBlockThreads(acc);
      if (size <= 32)
        return;
      if (blockThreadIdx < 32)
        warpPrefixScan(ws, blockThreadIdx, 0xffffffff);
      alpaka::block::sync::syncBlockThreads(acc);
      for (auto i = first + 32; i < size; i += blockDimension) {
        auto warpId = i / 32;
        c[i] += ws[warpId - 1];
      }
      alpaka::block::sync::syncBlockThreads(acc);
#else
      for (uint32_t i = 1; i < size; ++i)
        c[i] += c[i - 1];
#endif
    }
    };
    // same as above, may remove
    // limited to 32*32 elements....
    // struct blockPrefixScan {
    // ....... copy from ALPAKA_FN_INLINE...
    // };

    // limited to 1024*1024 elements....
    struct multiBlockPrefixScan {
    template <typename T, typename T_Acc>
    ALPAKA_FN_INLINE void operator()(const T_Acc& acc, T const* ci, T* co, int32_t size, int32_t* pc) {
      uint32_t const gridDimension(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
      uint32_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
      uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      
      auto&& ws = alpaka::block::shared::st::allocVar<T, 32>(acc);
      // first each block does a scan of size 1024; (better be enough blocks....)
      assert(gridDimension <= 1024);
      assert(blockDimension * gridDimension >= size);
      int off = blockDimension * gridBlockIdx;
      if (size - off > 0)
        blockPrefixScan(ci + off, co + off, std::min(int(blockDimension), size - off), ws);

      // count blocks that finished
      auto&& isLastBlockDone = alpaka::block::shared::st::allocVar<bool, 1>(acc);
      if (0 == blockThreadIdx) {
        auto value = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(acc, pc, 1);  // block counter
        isLastBlockDone = (value == (int(gridDimension) - 1));
      }

      alpaka::block::sync::syncBlockThreads(acc);

      if (!isLastBlockDone)
        return;

      assert(int(gridDimension)==*pc);

      // good each block has done its work and now we are left in last block

      // let's get the partial sums from each block
      T* psum = alpaka::block::shared::dyn::getMem<T, 1>(acc);

      // extern __shared__ T psum[];
      for (int i = blockThreadIdx, ni = gridDimension; i < ni; i += blockDimension) {
        auto j = blockDimension * i + blockDimension -1;
        psum[i] = (j < size) ? co[j] : T(0);
      }
      alpaka::block::sync::syncBlockThreads(acc);
      blockPrefixScan(psum, psum, gridDimension, ws);

      // now it would have been handy to have the other blocks around...
      int first = blockThreadIdx;                                 // + blockDimension * gridBlockIdx
      for (int i = first + blockDimension; i < size; i += blockDimension) {  //  *gridDimension) {
        auto k = i / blockDimension;                                     // block
        co[i] += psum[k - 1];
      }
    }
    };
  }  // namespace alpaka
}  // namespace cms

#endif  // HeterogeneousCore_AlpakaUtilities_interface_prefixScan_h
