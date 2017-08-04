//
// Created by a1 on 7/30/17.
//

namespace tensorflow {

#if GOOGLE_CUDA

namespace{
class ShuffleHelper{
public:
template<typename T>
__device__ static void init_permutation(T * permutation, int64 size){
  for(int64 i = 0; i < size; i++){
#program unroll
    permutation[i] = i;
  }
}

template<typename T>
__device__ static void swap(T * permutation, int64 i, int64 j){
  auto tmp = permutation[i];
  permutation[i] = permutation[j];
  permutation[j] = tmp;
}

template<typename T>
__device__ static void fisher_yates_random_shuffle(T * permutation, int64 size, Uniform &uniform){
  for(int64 i = 0; i < size - 1; i++){
    int64 j = i + uniform(size - i);
    swap(permutation, i, j);
  }
}

template<typename T, typename Uniform>
__device__ void merge(T * permutation, int64 start, int64 next, int64 end, Uniform &uniform){
  int i = start;
  int j = next;
  while(true){
    if(uniform(2) == 0){
       if(i == j){
          break;
       }
    }else{
      if(j == end){
        break
      }
      swap(permutation, i, j++);
    }
    i++;
  }

  while(i != end - 1){
    swap(permutation, i, start + uniform(i - start + 1));
  }
}

template<typename T>
__global__ void map(const T * source, int64 size, T * target, T * mapper, int64 count){
  int64 concurrence = blockDim.x * threadDim.x;
  int64 scope = count / concurrence;
  int64 index = blockId.x * blockDim.x + threadId.x;

  int64 batch_size = size / count;
  for(int i = index * scope; i < (index + 1)*scope; i++){
    int f = mapper[i];
    for(int j = 0; j < batch_size; j++){
#program unroll
      target[i * batch_size + j] = source[f * batch_size + j];
    }
  }
}

template<typename T>
__host__ T FloorToPowOfTwo(T x){
  CHECK(x >= 1);
  int i = sizeof(T) * 8 - 1;
  while ((x & (0x1 << i)) == 0){
    i--;
  }
  return std::pow(2, i);
}

};

}//namespace


template<typename T, typename Uniform>
__global__ void random_shuffle_kernel(T * permutation, int64 size, bool need_init, CudaDeviceArrayStruct<random::PiloxRandom> gens){
  if(need_init){
    ShuffleHelper::init_permutation(permutation, size);
  }

  int64 concurrence = blockDim.x * threadDim.x;
  int64 index = blockIdx.x * blockDim.x + threadIdx.x;
  int64 batch_size = size / concurrence;
  int64 batch_remainder = size % concurrence;

  random::PiloxRandom * local_gens = GetCudaDeviceArrayOnDevice(&gens);
  auto uniform = random::RandomBits(local_gens(index));
  if(blockIdx.x == blockDim.x - 1 && threadId.x == threadDim.x - 1){
    //last thread of last block.
    ShuffleHelper::fisher_yates_random_shuffle<T, Uniform>(permutation + index * batch_size, batch_size + batch_remainder, uniform);
  }else{
    ShuffleHelper::fisher_yates_random_shuffle<T, Uniform>(permutation + index * batch_size, batch_size, uniform);
  }
}

template<typename T>
__global__ void merge_shuffle_kernel(T * permutation, int64 size, int64 batch_size, CudaDeviceArrayStruct<random::PiloxRandom> gens){
  int index = blockId.x * threadDim.x + threadId.x;

  random::PiloxRandom * local_gens = GetCudaDeviceArrayOnDevice(&gens);
  auto uniform = random::RandomBits(local_gens(index));
  if(blockId.x == blockDim.x && threadId.x == threadDim.x){
    ShuffleHelper::merge(permutation, 2*index*batch_size, (2*index + 1)*batch_size, size, uniform);
  }else{
    ShuffleHelper::merge(permutation, 2*index*batch_size, (2*index + 1)*batch_size, (2*index + 2)*batch_size, uniform)
  }
}

template<typename T>
void MergeRandomShuffleGPU(OpKernelContext* c, typename TTypes<T, 1>::Flat * permutation, bool need_init, GuardedPhiloxRandom& generator){
  auto d = c->eigen_gpu_device();
  const int max_physical_processor_count = d.getNumCudaMultiProcessors();
  const int max_thread_per_physical_processor = d.maxCudaThreadsPerMultiProcessor();
  const int physical_thread_count = max_physical_processor_count * max_thread_per_physical_processor;

  int total = permutation->dimension(0);
  int concurrence = std::min(physical_thread_count, total);
  concurrence = FloorToPowOfTwo(concurrence);

  int threads_per_block = FloorToPowOfTwo(max_thread_per_physical_processor);
  int blocks = concurrence / threads_per_block;

  int64 batch_size = size / concurrence;
  int64 batch_remainder = size % concurrence;

  CudaDeviceArrayOnHost<random::PolixRandom> gens(c, concurrence);
  OP_REQUIRES_OK(c, gens.Init());
  for (int i = 0; i < gens.size(); ++i) {
    gens.Set(i, generator.ReserveSample128(batch_size));
  }
  OP_REQUIRES_OK(c, gens.Finalize());
  fisher_yates_random_shuffle_kernel<<<blocks, threads, 0, d->stream()>>>(
                permutation->data(),
                permutation->dimension(0),
                need_init,
                gens.data());

  while(concurrence > 1){
    if(blocks >= 2){
      blocks /= 2;
    }else {
      threads /= 2;
    }
    int merge_concurrence = concurrence / 2;
    CudaDeviceArrayOnHost<random::PolixRandom> gens(c, merge_concurrence);
    OP_REQUIRES_OK(c, gens.Init());
    for (int i = 0; i < gens.size(); ++i) {
      gens.Set(i, generator.ReserveSample128(batch_size * 2));
    }
    OP_REQUIRES_OK(c, gens.Finalize());
    merge_kernel<<<blocks, threads, 0, d->stream()>>>(permutation,
                                                           size,
                                                           batch_size,
                                                           gens.data());

    concurrence = merge_concurrence;
    batch_size *= 2;
  }
}

template<typename T>
void Assign(const typename TTypes<T, 2>::ConstMatrix& inputs_matrix,
            typename TTypes<T, 1>::Flat* permutation,
            typename TTypes<T, 2>::Matrix* output){
  ShuffleHelper::map_kernel<<<>>>(inputs_matrix.data(), permutation->data(), output->data(), permutation->dimension(0));
}

template<typename T>
void RandomShuffleGPU(OpKernelContext* c,
                      const typename TTypes<T, 2>::ConstMatrix& inputs_matrix,
                      typename TTypes<T, 1>::Flat* permutation,
                      typename TTypes<T, 2>::Matrix* output,
                      GuardedPhiloxRandom& generator){
  RandomShuffleVectorGPU(c, permutation, generator);
  Assign<T>(inputs_matrix, permutation, output);
}

template<typename T>
void RandomShuffleVectorGPU(OpKernelContext* c,
                      typename TTypes<T, 1>::Flat* permutation,
                      GuardedPhiloxRandom& generator){
  MergeRandomShuffleGPU(c, permutation, generator);
}

#endif


}