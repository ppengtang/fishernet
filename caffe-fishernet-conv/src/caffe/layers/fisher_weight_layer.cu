#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FisherWeightForwardGPU(const int nthreads,
          Dtype* top_data, const Dtype* weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = weight[index];
  }
}

template <typename Dtype>
void FisherWeightLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  int nthreads = N_;
  FisherWeightForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, weight);
}

template <typename Dtype>
__global__ void FisherWeightWeightBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    weight_diff[index] = top_diff[index];
  }
}

template <typename Dtype>
__global__ void FisherWeightBackwardGPU(const int nthreads, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = 0;
  }
}

template <typename Dtype>
void FisherWeightLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    // Gradient with respect to weight
    int nthreads = N_;
    FisherWeightWeightBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, weight_diff);
  }

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // Gradient with respect to bottom data
    int nthreads = bottom[0]->count();
    FisherWeightBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff);

  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(FisherWeightLayer);

}  // namespace caffe
