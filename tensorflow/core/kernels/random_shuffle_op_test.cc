/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <functional>
#include <memory>
#include <vector>

#include <thread>         // std::this_thread::sleep_for
#include <chrono>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil_2.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"


namespace tensorflow {

class RandomShuffleOpTestOnCpu : public OpsTestOnCpu{
};

TEST_F(RandomShuffleOpTestOnCpu, SimpleTest){
  TF_EXPECT_OK(NodeDefBuilder("random_shuffle", "RandomShuffle")
                      .Input(FakeInput(DT_INT32))
                      .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<int>(TensorShape({3,3,3}), {1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17,18, 19,20,21,22,23,24,25,26,27});
}

TEST_F(RandomShuffleOpTestOnCpu, LargeTest){
  std::vector<int> tensor_values;
  for(int i = 0; i < 1000000; i++){
    tensor_values.push_back(i);
  }

  TF_EXPECT_OK(NodeDefBuilder("random_shuffle", "RandomShuffle")
                    .Input(FakeInput(DT_INT32))
                    .Finalize(node_def()));

  TF_EXPECT_OK(InitOp());
  AddInputFromArray<int>(TensorShape({1000000}), tensor_values);
  TF_EXPECT_OK(RunOpKernel());
}


template<typename T>
void RandomShuffleHelper(int iters, int kDim1, int kDim2){
  testing::StopTiming();

  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  Tensor input(dt, TensorShape({kDim1, kDim2}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(
    NodeBuilder(g->NewName("n"), "RandomShuffle")
      .Input(test::graph::Constant(g, input))
      .Attr("T", dt)
      .Finalize(g, &node));

  testing::BytesProcessed(static_cast<int64>(iters) * ((kDim1 * kDim2)) * sizeof(T));
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
  testing::UseRealTime();
}

static void BM_RandomShuffleInt32Vector(int iters, int dim) {
  RandomShuffleHelper<int32>(iters, dim, 1);
}

static void BM_RandomShuffleInt32Matrix(int iters, int dim) {
  RandomShuffleHelper<int32>(iters, dim, 2);
}

static void BM_RandomShuffleInt32Cube(int iters, int dim){
  RandomShuffleHelper<int32>(iters, dim, 3);
}

BENCHMARK(BM_RandomShuffleInt32Vector)->Arg(1000)->Arg(100000)->Arg(1000000);
//BENCHMARK(BM_RandomShuffleInt32Matrix)->Arg(1000)->Arg(100000)->Arg(1000000);
//BENCHMARK(BM_RandomShuffleInt32Cube)->Arg(1000)->Arg(100000)->Arg(1000000);



template<typename T>
void RandomShuffleV2Helper(int iters, int kDim1, int kDim2){
  testing::StopTiming();

  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  Tensor input(dt, TensorShape({kDim1, kDim2}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(
    NodeBuilder(g->NewName("n"), "RandomShuffleV2")
      .Input(test::graph::Constant(g, input))
      .Attr("T", dt)
      .Finalize(g, &node));

  testing::BytesProcessed(static_cast<int64>(iters) * ((kDim1 * kDim2)) * sizeof(T));
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
  testing::UseRealTime();
}

static void BM_RandomShuffleV2Int32Vector(int iters, int dim) {
  RandomShuffleV2Helper<int32>(iters, dim, 1);
}

static void BM_RandomShuffleV2Int32Matrix(int iters, int dim) {
  RandomShuffleV2Helper<int32>(iters, dim, 2);
}

static void BM_RandomShuffleV2Int32Cube(int iters, int dim){
  RandomShuffleV2Helper<int32>(iters, dim, 3);
}

//BENCHMARK(BM_RandomShuffleV2Int32Vector)->Arg(1000)->Arg(100000)->Arg(1000000);
//BENCHMARK(BM_RandomShuffleV2Int32Matrix)->Arg(1000)->Arg(100000)->Arg(1000000);
//BENCHMARK(BM_RandomShuffleV2Int32Cube)->Arg(1000)->Arg(100000)->Arg(1000000);


template<typename T>
void RandomShuffleV3Helper(int iters, int kDim1, int kDim2){
  testing::StopTiming();

  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  Tensor input(dt, TensorShape({kDim1, kDim2}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(
    NodeBuilder(g->NewName("n"), "RandomShuffleV3")
      .Input(test::graph::Constant(g, input))
      .Attr("T", dt)
      .Finalize(g, &node));

  testing::BytesProcessed(static_cast<int64>(iters) * ((kDim1 * kDim2)) * sizeof(T));
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
  testing::UseRealTime();
}

static void BM_RandomShuffleV3Int32Vector(int iters, int dim) {
  RandomShuffleV3Helper<int32>(iters, dim, 1);
}

static void BM_RandomShuffleV3Int32Matrix(int iters, int dim) {
  RandomShuffleV3Helper<int32>(iters, dim, 2);
}

static void BM_RandomShuffleV3Int32Cube(int iters, int dim){
  RandomShuffleV3Helper<int32>(iters, dim, 3);
}

BENCHMARK(BM_RandomShuffleV3Int32Vector)->Arg(1000)->Arg(100000)->Arg(1000000);
//BENCHMARK(BM_RandomShuffleV3Int32Matrix)->Arg(1000)->Arg(100000)->Arg(1000000);
//BENCHMARK(BM_RandomShuffleV3Int32Cube)->Arg(1000)->Arg(100000)->Arg(1000000);

}
