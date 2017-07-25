//
// Created by a1 on 7/25/17.
//

#include "tensorflow/core/kernels/ops_testutil_2.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST_F(OpsTestOnCpu, ScopedStepContainer){
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                .Input(FakeInput(DT_STRING))
                .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<string>(TensorShape({}), {""});
  TF_EXPECT_OK(RunOpKernel());
  EXPECT_TRUE(step_container_ != nullptr);
}

#if GOOGLE_CUDA
TEST_F(OpsTestOnGpu, ScopedStepContainer){
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                .Input(FakeInput(DT_STRING))
                .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<string>(TensorShape({}), {""});
  TF_EXPECT_OK(RunOpKernel());
  EXPECT_TRUE(step_container_ != nullptr);
}
#endif


#ifdef TENSORFLOW_USE_SYCL
TEST_F(OpsTestOnSycl, ScopedStepContainer){
  TF_EXPECT_OK(NodeDefBuilder("identity", "Identity")
                .Input(FakeInput(DT_STRING))
                .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<string>(TensorShape({}), {""});
  TF_EXPECT_OK(RunOpKernel());
  EXPECT_TRUE(step_container_ != nullptr);
}
#endif


}