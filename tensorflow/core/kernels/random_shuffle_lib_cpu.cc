//
// Created by a1 on 7/30/17.
//

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/random_shuffle_lib.h"

namespace tensorflow{

template<typename Iter, class Random>
static inline void FisherYatesRandomShuffle(Iter first, Iter last, Random& uniform){
  if (first == last) return;
  const auto stop = last - 1;
  for (auto i = first; i != stop; ++i) {
    using std::iter_swap;
    iter_swap(i, i + uniform(last - i));
  }
}


template <typename Iter, typename Random>
void Merge(Iter first, Iter second, Iter last, Random& uniform){
  using std::iter_swap;
  auto i = first;
  auto j = second;
  auto k = last;
  auto flip = [&uniform]()->bool{return uniform(2) > 0;};

  while(true){
    if(flip()){
      if(i == j){
        break;
      }
    }else{
      if(j == k){
        break;
      }
      iter_swap(i, j++);
    }
    i++;
  }

  while(i != k){
    iter_swap(first + uniform(i - first + 1), i++);
  }
}

template<typename T>
T FloorToPowOfTwo(T x){
  CHECK(x >= 1);
  int i = sizeof(T) * 8 - 1;
  while ((x & (0x1 << i)) == 0){
    i--;
  }
  return std::pow(2, i);
}


template<typename Iter, class Random>
void MergeRandomShuffle(OpKernelContext * c, Iter first, Iter last, Random& uniform){
  auto worker_threads =
    c->device()->tensorflow_cpu_worker_threads();
  int max_parallelism = worker_threads->num_threads;
  CHECK(max_parallelism >= 1);

  if(max_parallelism == 1){
    FisherYatesRandomShuffle(first, last, uniform);
  }else {
    int total = last - first;
    const int min_block_size = 100;
    int needed_parallelism = total / min_block_size;
    int used_parallelism = std::min(needed_parallelism, max_parallelism);
    used_parallelism = FloorToPowOfTwo(used_parallelism);

    if(used_parallelism == 1){
      FisherYatesRandomShuffle(first, last, uniform);
    }else{
      std::vector<Iter> split_starts;
      BlockingCounter counter(used_parallelism - 1);
      for(int i = 0; i < used_parallelism - 1; i++){
        auto start = first + i * min_block_size;
        auto end = start + min_block_size;
        split_starts.push_back(start);
        worker_threads->workers->Schedule([&start, &end, &uniform, &counter](){
          FisherYatesRandomShuffle(start, end, uniform);
          counter.DecrementCount();
        });
      }

      //execute last block at current thread.
      split_starts.push_back(first + min_block_size * (used_parallelism - 1));
      FisherYatesRandomShuffle(split_starts[split_starts.size() - 1],  last, uniform);
      counter.Wait();

      while(split_starts.size() > 1){
        std::vector<Iter> new_split_starts;
        int merge_count = split_starts.size() / 2;
        BlockingCounter counter2(merge_count - 1);
        for(int i = 0; i < merge_count - 1; i++){
          int index = 2 * i;
          auto start = split_starts[index];
          auto start_next = split_starts[index + 1];
          auto end = split_starts[index + 2];

          new_split_starts.push_back(start);

          worker_threads->workers->Schedule([&start, &start_next, &end, &uniform, &counter2](){
            Merge(start, start_next, end, uniform);
            counter2.DecrementCount();
          });
        }

        int last_index = 2 * (merge_count - 1);
        new_split_starts.push_back(split_starts[last_index]);
        Merge(split_starts[last_index], split_starts[last_index + 1], last, uniform);
        counter2.Wait();
        split_starts = new_split_starts;
      }
    }
  }
}

template<typename T, typename Mapper>
void Assign(int64 size,
            const typename TTypes<T, 2>::ConstMatrix& input_matrix,
            typename TTypes<T, 2>::Matrix * output,
            Mapper& mapper){
  for(int64 i = 0; i < size; i++){
    output->template chip<0>(i) = input_matrix.template chip<0>(mapper(i));
  }
}


template<typename T>
void RandomShuffleCPU(OpKernelContext * c,
                      const typename TTypes<T, 2>::ConstMatrix& input_matrix,
                      typename TTypes<T, 2>::Matrix * output,
                      const std::function<int64(uint32)>& uniform){

  //auto random_thread_safe = [](int64 n)->int64 { return 0; };
  std::vector<int64> permutations(input_matrix.dimension(0));
  for (int64 i = 0; i < permutations.size(); i++) {
    permutations[i] = i;
  }

  int num_threads =
    c->device()->tensorflow_cpu_worker_threads()->num_threads;
  if(num_threads == 0 || input_matrix.dimension(0) <= 100){
    FisherYatesRandomShuffle(permutations.begin(), permutations.end(), uniform);
  }else {
    MergeRandomShuffle(c, permutations.begin(), permutations.end(), uniform);
  }

  std::cout<<"permutations:";
  for(auto item : permutations){
    std::cout<<","<<item;
  }
  std::cout<<std::endl;


  auto mapper = [&permutations](int64 i) ->int64 { return permutations[i]; };
  Assign<T>(input_matrix.dimension(0), input_matrix, output, mapper);
}


#define REGISTER(T)                                                         \
    template void RandomShuffleCPU<T>(                                      \
          OpKernelContext *,                                                \
          const typename TTypes<T, 2>::ConstMatrix&,                        \
          typename TTypes<T, 2>::Matrix * ,                                 \
          const std::function<int64(uint32)>& );
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER


}

