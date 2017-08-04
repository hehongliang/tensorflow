//
// Created by a1 on 8/3/17.
//

namespace tensorflow {
#if GOOGLE_CUDA


template<typename T>
void RandomShuffleGPU(OpKernelContext* c,
                      const typename TTypes<T, 2>::ConstMatrix& inputs_matrix,
                      typename TTypes<T, 2>::Matrix* output,
                      GuardedPhiloxRandom& generator);


#endif


}