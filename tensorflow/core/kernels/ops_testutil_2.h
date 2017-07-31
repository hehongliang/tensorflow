//
// Created by a1 on 7/25/17.
//

#ifndef TENSORFLOW_OPS_TESTUTIL_2_H
#define TENSORFLOW_OPS_TESTUTIL_2_H

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {

class OpsTestBase2 : public ::testing::Test {
public:
  //"gpu" or "cpu" or "sycl"
  OpsTestBase2(const char * device_type)
    : device_type_(device_type) {

    CHECK(DeviceFactory::GetFactory(device_type) != nullptr);
    device_.reset(
      DeviceFactory::NewDevice(device_type_.type_string(),
                               {},
                               "/job:a/replica:0/task:0"));
    CHECK(device_.get())<<"Could not create device:"<<device_type_.type_string();
  }

  ~OpsTestBase2() override { }

  void set_node_def(const NodeDef& node_def){ node_def_.CopyFrom(node_def); }

  NodeDef * node_def() {return &node_def_;}

  Status InitOp() {return InitOpWithGraphVersion(TF_GRAPH_DEF_VERSION);}

  Status InitOpWithGraphVersion(int graph_def_version) {
    Status status;
    kernel_ = CreateOpKernel(device_type_,
                             device_.get(),
                             allocator(),
                             node_def_,
                             graph_def_version,
                             &status);
    if(kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }

  template <typename T>
  void AddInput(const TensorShape& shape,
                std::function<T(int)> input_mapping){
    CHECK_GT(input_types_.size(), inputs_.size())<<
      "Adding more inputs than types; perhaps you need to call MakeOp";

    bool is_ref = IsRefType(input_types_[input_types_.size()]);
    Tensor * input =
      new Tensor(device_->GetAllocator(AllocatorAttributes()), DataTypeToEnum<T>::v(), shape);

    test::FillFn(input, input_mapping);
    tensors_.push_back(input);
    if(is_ref){
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]), DataTypeToEnum<T>::v());
      inputs_.push_back({&lock_for_refs_, input});
    }else{
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<T>::v());
      inputs_.push_back({nullptr, input});
    }
  }

  template <typename T>
  void AddInputFromArray(const TensorShape& shape, const gtl::ArraySlice<T>& data) {
    CHECK_GT(input_types_.size(), inputs_.size())
      <<"Adding more inputs than types; perhaps you need to call MakeOp";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor * input  = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                                 DataTypeToEnum<T>::v(), shape);

    LOG(INFO)<<"Tensor Allocator Name = " + input->AllocatorName();
    test::FillValues<T>(input, data);
    tensors_.push_back(input);
    if(is_ref){
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]), DataTypeToEnum<T>::v());
      inputs_.push_back({&lock_for_refs_, input});
    }else{
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<T>::v());
      inputs_.push_back({nullptr, input});
    }
  }

  template<typename T, typename SrcType>
  void AddInputFromList(const TensorShape& shape, std::initializer_list<SrcType> data){
    CHECK_GT(input_types_.size(), inputs_.size())
    << "Adding more inputs than types; perhaps you need to call MakeOp";
    bool is_ref = IsRefType(input_types_[inputs_.size()]);
    Tensor* input = new Tensor(device_->GetAllocator(AllocatorAttributes()),
                               DataTypeToEnum<T>::v(), shape);
    test::FillValues<T>(input, data);
    tensors_.push_back(input);
    if (is_ref) {
      CHECK_EQ(RemoveRefType(input_types_[inputs_.size()]),
               DataTypeToEnum<T>::v());
      inputs_.push_back({&lock_for_refs_, input});
    } else {
      CHECK_EQ(input_types_[inputs_.size()], DataTypeToEnum<T>::v());
      inputs_.push_back({nullptr, input});
    }
  }

  Status RunOpKernel() {
    context_.reset(nullptr);

    params_.reset(new OpKernelContext::Params);
    params_.get()->device = device_.get();
    params_.get()->frame_iter = FrameAndIter(0, 0);
    params_.get()->inputs = &inputs_;
    params_.get()->op_kernel = kernel_.get();
    step_container_.reset(new ScopedStepContainer(0, [](const string& ) {}));
    params_->step_container = step_container_.get();
    std::vector<AllocatorAttributes> attrs;
    SetOutputAttrs(params_.get(), &attrs);
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_.get()->slice_reader_cache = &slice_reader_cache_wrapper;
    params_.get()->resource_manager = device_.get()->resource_manager();

    context_.reset(new OpKernelContext(params_.get()));
    device_->Compute(kernel_.get(), context_.get());

    return context_->status();
  }

  const Tensor& GetInput(int input_index) const {
    CHECK_LT(input_index, context_->num_inputs());
    CHECK(!IsRefType(context_->input_dtype(input_index)));
    return context_->input(input_index);
  }

  TensorValue mutable_input(int input_index) {
    CHECK_LT(input_index, inputs_.size());
    return inputs_[input_index];
  }

  Tensor * GetOutput(int output_index) {
    CHECK_LT(output_index, context_->num_outputs());
    return context_->mutable_output(output_index);
  }

  Allocator * allocator() {
    return device_->GetAllocator(AllocatorAttributes());
  }

  const DataTypeVector& output_types() const { return kernel_->output_types(); }


private:
  void SetOutputAttrs(OpKernelContext::Params* params,
                      std::vector<AllocatorAttributes> * attrs){
    attrs->clear();
    for(int index = 0; index < params->op_kernel->num_outputs();index++){
      AllocatorAttributes attr;
      const bool on_host =
        (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
      attr.set_on_host(on_host);
      attrs->push_back(attr);
    }
    params->output_attr_array = gtl::vector_as_array(attrs);
  }

protected:
  std::unique_ptr<Device> device_;

  std::unique_ptr<OpKernel> kernel_;

  std::unique_ptr<ScopedStepContainer> step_container_;

  NodeDef node_def_;

  DataTypeVector input_types_;
  DeviceType device_type_;

  mutex lock_for_refs_;  // Used as the Mutex for inputs added as refs

  gtl::InlinedVector<TensorValue, 4> inputs_;
  // Owns Tensors.
  std::vector<Tensor*> tensors_;

  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<OpKernelContext> context_;

private:
  TF_DISALLOW_COPY_AND_ASSIGN(OpsTestBase2);
};

class OpsTestOnCpu : public OpsTestBase2{
public:
  OpsTestOnCpu() : OpsTestBase2(DEVICE_CPU){}
  ~OpsTestOnCpu() override {}
};

class OpsTestOnGpu : public OpsTestBase2{
public:
  OpsTestOnGpu() : OpsTestBase2(DEVICE_GPU){}
  ~OpsTestOnGpu() override {}
};

class OpsTestOnSycl : public OpsTestBase2 {
public:
  OpsTestOnSycl() : OpsTestBase2(DEVICE_SYCL) {}
  ~OpsTestOnSycl() override {}
};


}


#endif //TENSORFLOW_OPS_TESTUTIL_2_H
