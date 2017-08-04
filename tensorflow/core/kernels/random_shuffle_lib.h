//
// Created by a1 on 7/29/17.
//

#ifndef TENSORFLOW_RANDOM_SHUFFLE_LIB_H
#define TENSORFLOW_RANDOM_SHUFFLE_LIB_H


namespace tensorflow{

template<typename T>
void RandomShuffleCPU(OpKernelContext * c,
                      const typename TTypes<T, 2>::ConstMatrix& input_matrix,
                      typename TTypes<T, 2>::Matrix * output,
                      GuardedPhiloxRandom& generator);


#if GOOGLE_CUDA
template<typename T>
void RandomShuffleGPU(OpKernelContext* c,
                      const typename TTypes<T, 2>::ConstMatrix& inputs_matrix,
                      typename TTypes<T, 1>::Flat& permutation,
                      typename TTypes<T, 2>::Matrix* output,
                      GuardedPhiloxRandom& generator);
#endif




}


#endif //TENSORFLOW_RANDOM_SHUFFLE_LIB_H
