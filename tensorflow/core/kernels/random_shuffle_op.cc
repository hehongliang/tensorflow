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

// See docs in ../ops/random_ops.cc.

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include "tensorflow/core/kernels/random_shuffle_lib.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif // TENSORFLOW_USE_SYCL


// TODO(irving): If performance is critical, generate output directly instead
// of an in-place shuffle using a pseudorandom permutation like
//
//   https://github.com/otherlab/geode/blob/master/geode/random/permute.cpp
//
// This is probably also the right thing if we want a GPU version of shuffling.

// We use our own version of std::random_shuffle to guarantee that exactly
// size - 1 samples are used.
template <class Iter, class Random>
static inline void RandomShuffle(Iter first, Iter last, Random& uniform) {
  if (first == last) return;
  const auto stop = last - 1;
  for (auto i = first; i != stop; ++i) {
    using std::iter_swap;
    iter_swap(i, i + uniform(last - i));
  }
}

template <class IntT, class InT, class OutT, class Random>
static void IndexedShuffle(const int64 size, const InT& input_mat,
                           OutT output_mat, Random& uniform) {
  std::vector<IntT> permutation(size);
  for (IntT i = 0; i < size; i++) {
    permutation[i] = i;
  }
  RandomShuffle(permutation.begin(), permutation.end(), uniform);
  for (IntT i = 0; i < size; i++) {
    output_mat.template chip<0>(i) = input_mat.template chip<0>(permutation[i]);
  }
}

namespace my_dbg{

static void PrintTensor(const Tensor& t){
  auto dims = t.dims();
  auto allocatorName = t.AllocatorName();
  auto dtype = t.dtype();
  std::cout
  <<" origin_tensor:"
  <<" dims="<<dims
  <<" AllocatorName="<<allocatorName
  <<" dtype="<<dtype
  <<" dimensions=(";
  TensorShape shape = t.shape();
  for(int i = 0; i < shape.dims(); i++){
    std::cout<<shape.dim_size(i);
    if(i < shape.dims() - 1){
      std::cout<<",";
    }
  }
  std::cout<<")"<<std::endl;
}

template<typename T>
class DataToString;

template<>
struct  DataToString<int> {
public:
  static string trans(int t){
    return std::to_string(t);
  }
};

template<typename T>
struct DataToString{
public:
  static string trans(const T& t){
    DataType dtype = DataTypeToEnum<T>::value;
    return string("not support to string, dtype=") + DataToString<int>::trans(static_cast<int>(dtype));
  }
};


template<typename T, int NDIMS>
static void PrintEigenTensor(typename TTypes<T, NDIMS>::ConstTensor  t){

  std::cout<<std::endl;
  std::cout<<"DebugPrintEigenTensor"<<std::endl;

  std::cout
  <<" eigen_tensor base info:"
  <<" rank="<<t.rank()
  <<" size="<<t.size()
  <<" dimensions=(";
  auto dimensions = t.dimensions();
  for(int i = 0; i < dimensions.size(); i++){
    std::cout<<dimensions[i];
    if(i < dimensions.size() - 1) {
      std::cout<<",";
    }
  }
  std::cout<<")"<<std::endl;


  std::cout<<" al_elements=(";
  for(int i = 0; i < t.size(); i++){
    std::cout<<DataToString<T>::trans(t(i));
    if(i < t.size() - 1){
      std::cout<<",";
    }
  }
  std::cout<<")"<<std::endl;

}

template <typename T>
static void PrintChip(T & evaluator){
  std::cout<<std::endl;
  std::cout<<"DebugPrintEigenTensor"<<std::endl;
  auto dimensions = evaluator.dimensions();
  std::cout
  <<" chip_dims="<<dimensions.rank()
  <<" dimensions=(";


  int num_size = 1;
  for(int i = 0; i < dimensions.size(); i++){
    std::cout<<dimensions[i];
    num_size *= dimensions[i];
    if(i < dimensions.size() - 1) {
      std::cout<<",";
    }
  }
  std::cout<<")"<<std::endl;


  for(int i = 0; i < num_size; i++){
    std::cout<<DataToString<typename T::CoeffReturnType>::trans(evaluator.coeff(i));
    if(i < num_size - 1){
      std::cout<<",";
    }
  }
  std::cout<<std::endl;
}


template <typename T>
static void DebugPrintTensor(const Tensor& t){
  std::cout<<std::endl;
  std::cout<<"DebugPrintTensor"<<std::endl;

  std::cout<<" T ="<<DataTypeToEnum<T>::v()<<std::endl;

  PrintTensor(t);
  //PrintEigenTensor<T, 1>(t.flat<T>());
  /*PrintEigenTensor<T, 1>(t.shaped<T, 1>({t.NumElements()}));
  PrintEigenTensor<T, 2>(t.shaped<T, 2>({3, t.NumElements() / 3}));
  PrintEigenTensor<T, 2>(t.flat_inner_dims<T, 2>());
  PrintEigenTensor<T, 2>(t.flat_outer_dims<T, 2>());
  PrintEigenTensor<T, 3>(t.tensor<T, 3>());

  typename TTypes<T, 3>::ConstTensor eigen_t = t.tensor<T, 3>();
  PrintEigenTensor<T, 3>(eigen_t);



  const Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor> chip00 = eigen_t.template chip<0>(0);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator01(chip00, Eigen::DefaultDevice());
  PrintChip(evaluator01);

  const Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor> chip01 = eigen_t.template chip<0>(1);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator02(chip01, Eigen::DefaultDevice());
  PrintChip(evaluator02);

  const Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor> chip02 = eigen_t.template chip<0>(2);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator03(chip02, Eigen::DefaultDevice());
  PrintChip(evaluator03);

  //
  //
  const Eigen::TensorChippingOp<1, const typename TTypes<T, 3>::ConstTensor> chip10 = eigen_t.template chip<1>(0);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<1, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator10(chip10, Eigen::DefaultDevice());
  PrintChip(evaluator10);


  const Eigen::TensorChippingOp<1, const typename TTypes<T, 3>::ConstTensor> chip11 = eigen_t.template chip<1>(1);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<1, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator11(chip11, Eigen::DefaultDevice());
  PrintChip(evaluator11);

  const Eigen::TensorChippingOp<1, const typename TTypes<T, 3>::ConstTensor> chip12 = eigen_t.template chip<1>(2);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<1, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator12(chip12, Eigen::DefaultDevice());
  PrintChip(evaluator12);


  //
  //
  const Eigen::TensorChippingOp<2, const typename TTypes<T, 3>::ConstTensor> chip20 = eigen_t.template chip<2>(0);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<2, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator20(chip20, Eigen::DefaultDevice());
  PrintChip(evaluator20);

  const Eigen::TensorChippingOp<2, const typename TTypes<T, 3>::ConstTensor> chip21 = eigen_t.template chip<2>(1);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<2, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator21(chip21, Eigen::DefaultDevice());
  PrintChip(evaluator21);

  const Eigen::TensorChippingOp<2, const typename TTypes<T, 3>::ConstTensor> chip22 = eigen_t.template chip<2>(2);
  Eigen::TensorEvaluator<const Eigen::TensorChippingOp<2, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice>
    evaluator22(chip22, Eigen::DefaultDevice());
  PrintChip(evaluator22);*/





  //Eigen::internal::TensorExecutor<typename Eigen::TensorChippingOp<0, const typename TTypes<T, 3>::ConstTensor>, Eigen::DefaultDevice, false>::run(chip, Eigen::DefaultDevice());

  //const Tensor& slice = t.Slice(4, 8);
  //PrintTensor(slice);
  //PrintEigenTensor<T, 1>(slice.vec<T>());
  //PrintEigenTensor<T, 2>(slice.shaped<T, 2>({2, 2}));
}

void DebugPhiloxRandom(GuardedPhiloxRandom& random){
  std::cout<<std::endl;
  std::cout<<"DebugPhiloxRandom"<<std::endl;

  int samples = 10;
  auto local_gen = random.ReserveSamples32(samples);
  auto randoms = local_gen();
  for(int i = 0; i < randoms.size(); i++){
    std::cout<<randoms[i];
    std::cout<<",";
  }

  auto randoms1 = local_gen();
  for(int i = 0; i < randoms1.size(); i++){
    std::cout<<randoms1[i];
    std::cout<<",";
  }
}

}


template <typename T>
class RandomShuffleOp : public OpKernel {
 public:
  explicit RandomShuffleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    if (input.NumElements() <= 1 || input.dim_size(0) <= 1) {
      // No shuffling is required, so copy input directly to output
      context->set_output(0, input);
    } else {
      // Reserve enough random samples for shuffling
      const int64 size = input.dim_size(0);
      const int64 samples = size - 1;
      auto local_gen = generator_.ReserveSamples32(samples);
      random::SingleSampleAdapter<random::PhiloxRandom> single(&local_gen);
      const auto uniform = [&single](uint32 n) { return single() % n; };

      if (input.dims() == 1) {
        // For 1D data, copy and then shuffle in place
        context->set_output(0, tensor::DeepCopy(input));
        auto vec = context->mutable_output(0)->vec<T>();
        RandomShuffle(vec.data(), vec.data() + size, uniform);
      } else {
        // For >= 2D, shuffle indices and then copy across
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        const auto input_mat = input.flat_outer_dims<T>();
        auto output_mat = output->flat_outer_dims<T>();
        if (size < kint32max) {
          IndexedShuffle<int32>(size, input_mat, output_mat, uniform);
        } else {
          IndexedShuffle<int64>(size, input_mat, output_mat, uniform);
        }
      }
    }

  }

 private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("RandomShuffle").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RandomShuffleOp<T>);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER


//
// RandomShuffleV2Op
//
template <typename T>
class RandomShuffleV2Op : public OpKernel {
public:
  explicit RandomShuffleV2Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    if (input.NumElements() <= 1 || input.dim_size(0) <= 1) {
      // No shuffling is required, so copy input directly to output
      context->set_output(0, input);
    } else {
      // Reserve enough random samples for shuffling
      const int64 size = input.dim_size(0);
      const int64 samples = size - 1;
      auto local_gen = generator_.ReserveSamples32(samples);
      random::RandomBitsAdapter<random::PhiloxRandom> randomBits(&local_gen);
      const auto uniform = [&randomBits](uint32 n) { return randomBits(n); };

      if (input.dims() == 1) {
        // For 1D data, copy and then shuffle in place
        context->set_output(0, tensor::DeepCopy(input));
        auto vec = context->mutable_output(0)->vec<T>();
        RandomShuffle(vec.data(), vec.data() + size, uniform);
      } else {
        // For >= 2D, shuffle indices and then copy across
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        const auto input_mat = input.flat_outer_dims<T>();
        auto output_mat = output->flat_outer_dims<T>();
        if (size < kint32max) {
          IndexedShuffle<int32>(size, input_mat, output_mat, uniform);
        } else {
          IndexedShuffle<int64>(size, input_mat, output_mat, uniform);
        }
      }
    }

  }

private:
  GuardedPhiloxRandom generator_;
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("RandomShuffleV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RandomShuffleV2Op<T>);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER



//
// RandomShuffleV3Op
//
template <typename Device, typename T>
class RandomShuffleV3Op : public OpKernel {
public:
  explicit RandomShuffleV3Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, generator_.Init(context));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    //const int64 size = input.dim_size(0);

    if (input.NumElements() <= 1 || input.dim_size(0) <= 1) {
      // No shuffling is required, so copy input directly to output
      context->set_output(0, input);
    } else {
      if (input.dims() == 1) {
        // For 1D data, copy and then shuffle in place
        context->set_output(0, tensor::DeepCopy(input));
        auto vec = context->mutable_output(0)->vec<T>();
#if GOOGLE_CUDA
        if(std::is_same<Device, GPUDevice>::value){
          RandomShuffleVectorGPU<T>(context, &vec, generator_);
          return;
        }
#endif
        RandomShuffleVectorCPU<T>(context, &vec, generator_);
      } else {
        // For >= 2D, shuffle indices and then copy across
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input.shape(), &output));
        const auto input_mat = input.flat_outer_dims<T>();
        auto output_mat = output->flat_outer_dims<T>();

#if GOOGLE_CUDA
        if(std::is_same<Device, GPUDevice>::value){
        Tensor permutation;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<int64>::v(),
                                             TensorShape({input.dim_size(0)}),
                                             &permutation));
	auto eigen_vec = permutation.vec<int64>();
        RandomShuffleGPU<T>(context, input_mat, &eigen_vec, &output_mat, generator_);
        return ;
      }
#endif
        RandomShuffleCPU<T>(context, input_mat, &output_mat, generator_);
      }
    }
  }


private:
  GuardedPhiloxRandom generator_;
};


#define REGISTER(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("RandomShuffleV3").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      RandomShuffleV3Op<CPUDevice, T>);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER


#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("RandomShuffleV3").Device(DEVICE_GPU).TypeConstraint<T>("T"),                 \
      RandomShuffleV3Op<GPUDevice, T>);
TF_CALL_POD_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif




}  // namespace tensorflow
