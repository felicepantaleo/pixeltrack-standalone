#include <iostream>

#include "HeterogeneousCore/AlpakaCore/alpakaConfig.h"
#include "HeterogeneousCore/AlpakaUtilities/interface/prefixScan.h"

using namespace cms::alpaka;

template <typename T>
struct format_traits {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %d %d\n";
};

template <>
struct format_traits<float> {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %f %f\n";
};

struct testPrefixScan {
template <typename T, typename T_Acc>
ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t size) const {
  auto&& T ws = alpaka::block::shared::st::allocVar<T, 32>(acc);
  auto&& T c = alpaka::block::shared::st::allocVar<T, 1024>(acc);
  auto&& T co = alpaka::block::shared::st::allocVar<T, 1024>(acc);

  uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
  uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

  auto first = blockThreadIdx;
  for (auto i = first; i < size; i += blockDimension)
    c[i] = 1;
  alpaka::block::sync::syncBlockThreads(acc);

  blockPrefixScan(c, co, size, ws);
  blockPrefixScan(c, size, ws);

  assert(1 == c[0]);
  assert(1 == co[0]);
  for (auto i = first + 1; i < size; i += blockDimension) {
    if (c[i] != c[i - 1] + 1)
      printf(format_traits<T>::failed_msg, size, i, blockDimension, c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}
};

struct testWarpPrefixScan {
template <typename T, typename T_Acc>
ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t size) const {
  assert(size <= 32);
  auto&& T c = alpaka::block::shared::st::allocVar<T, 1024>(acc);
  auto&& T co = alpaka::block::shared::st::allocVar<T, 1024>(acc);

  uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
  uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
  auto i = blockThreadIdx;
  c[i] = 1;
  alpaka::block::sync::syncBlockThreads(acc);
  auto laneId = blockThreadIdx & 0x1f;
  warpPrefixScan(laneId, c, co, i, 0xffffffff);
  warpPrefixScan(laneId, c, i, 0xffffffff);
  alpaka::block::sync::syncBlockThreads(acc);

  assert(1 == c[0]);
  assert(1 == co[0]);
  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      printf(format_traits<T>::failed_msg, size, i, blockDim.x, c[i], c[i - 1]);
    assert(c[i] == c[i - 1] + 1);
    assert(c[i] == i + 1);
    assert(c[i] = co[i]);
  }
}
};

struct init {
template <typename T_Acc>
ALPAKA_FN_ACC void operator()(const T_Acc& acc, uint32_t *v, uint32_t val, uint32_t n) const {
  uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
  uint32_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
  uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
  auto i = gridBlockIdx * blockDimension + blockThreadIdx;
  if (i < n)
    v[i] = val;
  if (i == 0)
    printf("init\n");
}
};

struct verify {
template <typename T_Acc>
ALPAKA_FN_ACC void operator()(const T_Acc& acc,
__global__ void verify(uint32_t const *v, uint32_t n) const {
  uint32_t const blockDimension(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
  uint32_t const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u]);
  uint32_t const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
  auto i = gridBlockIdx * blockDimension + blockThreadIdx;  if (i < n)
    assert(v[i] == i + 1);
  if (i == 0)
    printf("verify\n");
}
};

int main() {
  const DevHost host(alpaka::pltf::getDevByIdx<PltfHost>(0u));
  const DevAcc device(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
  const Vec size(1u);

  Queue queue(device);

  Vec elementsPerThread(Vec::all(1));
  Vec threadsPerBlock(Vec::all(512));
  Vec blocksPerGrid(Vec::all((input.wordCounter + 512 - 1) / 512));
#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED || ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || ALPAKA_ACC_CPU_BT_OMP4_ENABLED
  // on the GPU, run with 512 threads in parallel per block, each looking at a single element
  // on the CPU, run serially with a single thread per block, over 512 elements
  std::swap(threadsPerBlock, elementsPerThread);
#endif
  
  const WorkDiv workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);
  std::cout << "blocks per grid: " << blocksPerGrid << ", threads per block: " << threadsPerBlock << ", elements per thread: " << elementsPerThread << std::endl;

  std::cout << "warp level" << std::endl;
  // std::cout << "warp 32" << std::endl;

  alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>(workDiv, testWarpPrefixScan(), 32);
  alpaka::wait::wait(queue);

  alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>(workDiv, testWarpPrefixScan(), 16);
  alpaka::wait::wait(queue);

  alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>(workDiv, testWarpPrefixScan(), 5);
  alpaka::wait::wait(queue);
  
  std::cout << "block level" << std::endl;
  for (int bs = 32; bs <= 1024; bs += 32) {
    // std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= 1024; ++j) {
      // running kernel with 1 block, bs threads per block, 1 element per thread
      alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>({Vec::all(1),Vec::all(bs),Vec::all(1)}, testPrefixScan<uint16_t>(), j);
      alpaka::wait::wait(queue);
      alpaka::queue::enqueue(
            queue,
            alpaka::kernel::createTaskKernel<Acc>({Vec::all(1),Vec::all(bs),Vec::all(1)}, testPrefixScan<float>(), j);
      alpaka::wait::wait(queue);
    }
  }
  alpaka::wait::wait(queue);

  int num_items = 200;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    std::cout << "multiblock" << std::endl;
    // Declare, allocate, and initialize device-accessible pointers for input and output
    num_items *= 10;
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    auto input_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, num_items * sizeof(uint32_t));
    uint32_t* input_d = alpaka::mem::view::getPtrNative(input_dBuf);

    auto output1_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, num_items * sizeof(uint32_t));
    uint32_t* output1_d = alpaka::mem::view::getPtrNative(output1_dBuf);

    auto output2_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, num_items * sizeof(uint32_t));
    uint32_t* output2_d = alpaka::mem::view::getPtrNative(output2_dBuf);

    auto nthreads = 256;
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    init<<<nblocks, nthreads, 0>>>(d_in, 1, num_items);


    alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>({Vec::all(nblocks),Vec::all(nthreads),Vec::all(1)}, init(), input_d, 1, num_items);
    alpaka::wait::wait(queue);
    // the block counter
    int32_t *d_pc;
    // cudaCheck(cudaMalloc(&d_pc, sizeof(int32_t)));
    // cudaCheck(cudaMemset(d_pc, 0, 4));
    alpaka::mem::view::set(queue, pc_dBuf, 0,4);
    auto pc_dBuf = alpaka::mem::buf::alloc<uint32_t, Idx>(device, sizeof(uint32_t));
    uint32_t* pc_d = alpaka::mem::view::getPtrNative(pc_dBuf);


    nthreads = 512+256;
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "launch multiBlockPrefixScan " << num_items <<' '<< nblocks << std::endl;
    // multiBlockPrefixScan<<<nblocks, nthreads, 4*nblocks>>>(input_d, output1_d, num_items, d_pc);
    alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>({Vec::all(nblocks),Vec::all(nthreads),Vec::all(1)}, multiBlockPrefixScan(), input_d, output1_d, num_items, pc_d);
    alpaka::wait::wait(queue);


    cudaCheck(cudaGetLastError());
    // verify<<<nblocks, nthreads, 0>>>(output1_d, num_items);
        alpaka::queue::enqueue(
          queue,
          alpaka::kernel::createTaskKernel<Acc>({Vec::all(nblocks),Vec::all(nthreads),Vec::all(1)}, verify(), output1_d, num_items);
    alpaka::wait::wait(queue);


  }  // ksize
  return 0;
}
