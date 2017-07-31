//
// Created by a1 on 7/30/17.
//

namespace tensorflow {

#if GOOGLE_CUDA

template<>
__global__ void FisherYatesRandomShuffle(){
}

template<>
__global__ void MergeRandomShuffle(){
}



template<typename T>
void RandomShuffleGPU(OpKernelContext* c,
                      const typename TTypes<T, 2>::ConstMatrix& inputs_matrix,
                      typename TTypes<T, 2>::Matrix* output){


}

#endif


}