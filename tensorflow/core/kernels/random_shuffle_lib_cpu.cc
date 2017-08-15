//
// Created by a1 on 7/30/17.
//

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include "tensorflow/core/kernels/random_shuffle_lib.h"

namespace tensorflow{

namespace {


template<typename Iter, class Random>
static inline void FisherYatesRandomShuffle(Iter first, Iter last, Random &uniform) {
  if (first == last) return;

  const auto stop = last - 1;
  for (auto i = first; i != stop; ++i) {
    using std::iter_swap;
    iter_swap(i, i + uniform(last - i));
  }
}


template<typename Iter, typename Random>
void Merge(Iter first, Iter second, Iter last, Random &uniform) {
  using std::iter_swap;
  auto i = first;
  auto j = second;
  auto k = last;

  while (true) {
    if (uniform(2) > 0) {
      if (i == j) {
        break;
      }
    } else {
      if (j == k) {
        break;
      }
      iter_swap(i, j++);
    }
    i++;
  }

  while (i != k) {
    iter_swap(first + uniform(i - first + 1), i);
    i++;
  }
}

template<typename T>
T FloorToPowOfTwo(T x) {
  CHECK(x >= 1);
  int i = sizeof(T) * 8 - 1;
  while ((x & (0x1 << i)) == 0) {
    i--;
  }
  return std::pow(2, i);
}

template<typename Iter, class Random>
void MergeRandomShuffle(OpKernelContext *c, Iter first, Iter last, Random &generator) {

  auto worker_threads =
    c->device()->tensorflow_cpu_worker_threads();
  int max_parallelism = worker_threads->num_threads;
  CHECK(max_parallelism >= 1);

  if (max_parallelism == 1) {
    int samples = last - first;
    auto local_gen = generator.ReserveSamples32(samples);
    random::RandomBitsAdapter <random::PhiloxRandom> randomBits(&local_gen);
    FisherYatesRandomShuffle(first, last, randomBits);
  } else {
    int total = last - first;
    const int min_block_size = 1000;
    int needed_parallelism = total / min_block_size;
    int used_parallelism = std::min(needed_parallelism, max_parallelism);
    used_parallelism = FloorToPowOfTwo(used_parallelism);

    //std::cout<<" used_parallelism="<<used_parallelism;

    if (used_parallelism == 1) {
      int samples = last - first;
      auto local_gen = generator.ReserveSamples32(samples);
      random::RandomBitsAdapter <random::PhiloxRandom> randomBits(&local_gen);
      FisherYatesRandomShuffle(first, last, randomBits);
    } else {
      std::vector <Iter> split_starts;
      BlockingCounter counter(used_parallelism - 1);

      int used_block_size = total / used_parallelism;
      for (int i = 0; i < used_parallelism - 1; i++) {
        auto start = first + i * used_block_size;
        auto end = start + used_block_size;
        split_starts.push_back(start);

        int samples = end - start;
        auto local_gen = generator.ReserveSamples32(samples);
        worker_threads->workers->Schedule([start, end, &counter, local_gen]() mutable {
          random::RandomBitsAdapter <random::PhiloxRandom> randomBits(&local_gen);
          FisherYatesRandomShuffle(start, end, randomBits);
          counter.DecrementCount();
        });
      }

      //execute last block at current thread.
      split_starts.push_back(first + used_block_size * (used_parallelism - 1));

      int samples = last - split_starts[split_starts.size() - 1];
      auto local_gen = generator.ReserveSamples32(samples);
      random::RandomBitsAdapter <random::PhiloxRandom> randomBits(&local_gen);
      FisherYatesRandomShuffle(split_starts[split_starts.size() - 1], last, randomBits);
      counter.Wait();

      while (split_starts.size() > 1) {
        std::vector <Iter> new_split_starts;
        int merge_count = split_starts.size() / 2;
        BlockingCounter counter2(merge_count - 1);
        for (int i = 0; i < merge_count - 1; i++) {
          int index = 2 * i;
          auto start = split_starts[index];
          auto start_next = split_starts[index + 1];
          auto end = split_starts[index + 2];

          new_split_starts.push_back(start);

          int samples = end - start;
          auto local_gen = generator.ReserveSamples32(samples);
          worker_threads->workers->Schedule([start, start_next, end, local_gen, &counter2]() mutable {
            random::RandomBitsAdapter <random::PhiloxRandom> randomBits(&local_gen);
            Merge(start, start_next, end, randomBits);
            counter2.DecrementCount();
          });
        }

        int last_index = 2 * (merge_count - 1);
        new_split_starts.push_back(split_starts[last_index]);

        int samples = last - split_starts[last_index];
        auto local_gen = generator.ReserveSamples32(samples);
        random::RandomBitsAdapter <random::PhiloxRandom> randomBits(&local_gen);
        Merge(split_starts[last_index], split_starts[last_index + 1], last, randomBits);
        counter2.Wait();
        split_starts = new_split_starts;
      }
    }
  }
}

template<typename T, typename Mapper>
void Assign(int64 size,
            const typename TTypes<T, 2>::ConstMatrix &input_matrix,
            typename TTypes<T, 2>::Matrix *output,
            Mapper &mapper) {
  for (int64 i = 0; i < size; i++) {
    auto j = mapper(i);
    output->template chip<0>(i) = input_matrix.template chip<0>(j);
  }
}

template<typename Iter>
void RandomShuffleVector(OpKernelContext * c,
                            Iter first,
                            Iter last,
                            GuardedPhiloxRandom& generator){
  int num_threads =
    c->device()->tensorflow_cpu_worker_threads()->num_threads;
  int samples = last - first;
  if(num_threads == 0 || samples <= 1000){
    auto local_gen = generator.ReserveSamples32(samples);
    random::RandomBitsAdapter<random::PhiloxRandom> randomBits(&local_gen);
    FisherYatesRandomShuffle(first, last, randomBits);
  }else {
    MergeRandomShuffle(c, first, last, generator);
  }
}


} //namespace



template<typename T>
void RandomShuffleVectorCPU(OpKernelContext * c,
                            typename TTypes<T, 1>::Vec * permutation,
                            GuardedPhiloxRandom& generator){
  RandomShuffleVector(c, permutation->data(), permutation->data() + permutation->size(), generator);
}

#define REGISTER(T)                                                                   \
  template void RandomShuffleVectorCPU<T>(OpKernelContext * c,                        \
                                          typename TTypes<T, 1>::Vec * permutation,   \
                                          GuardedPhiloxRandom& generator);
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER



template<typename T>
void RandomShuffleCPU(OpKernelContext * c,
                      const typename TTypes<T, 2>::ConstMatrix& input_matrix,
                      typename TTypes<T, 2>::Matrix * output,
                      GuardedPhiloxRandom& generator){
  std::vector<int64> permutations(input_matrix.dimension(0));
  for (int64 i = 0; i < permutations.size(); i++) {
    permutations[i] = i;
  }

  RandomShuffleVector(c, permutations.begin(), permutations.end(), generator);
  auto mapper = [&permutations](int64 i) ->int64 { return permutations[i]; };
  Assign<T>(input_matrix.dimension(0), input_matrix, output, mapper);
}


#define REGISTER(T)                                                         \
    template void RandomShuffleCPU<T>(                                      \
          OpKernelContext *,                                                \
          const typename TTypes<T, 2>::ConstMatrix&,                        \
          typename TTypes<T, 2>::Matrix * ,                                 \
          GuardedPhiloxRandom& );
TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER


}

