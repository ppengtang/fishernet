#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FisherSumForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* bottom_data_sigma,
          Dtype* top_data, const int N_, const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = (index % (N_ * inner_num_)) / inner_num_;
    top_data[index] = bottom_data[index] + bottom_data_sigma[n];
  }
}

template <typename Dtype>
void FisherSumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data_sigma = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int inner_num_ = bottom[0]->count() / (N_ * outer_num_);
  const int nthreads = outer_num_ * N_ * inner_num_;
  FisherSumForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, bottom_data_sigma, top_data, 
                                N_, inner_num_);
}

template <typename Dtype>
__global__ void FisherSumBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = top_diff[index];
  }
}

template <typename Dtype>
__global__ void FisherSumSigmaBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff_sigma, 
          const int outer_num_, const int N_, const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index;
    bottom_diff_sigma[n] = 0;
    for (int i = 0; i < outer_num_; i++) {
      for (int j = 0; j < inner_num_; j++) {
        bottom_diff_sigma[n] += top_diff[i * N_ * inner_num_ + n * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
void FisherSumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff_sigma = bottom[1]->mutable_gpu_diff();
    const int inner_num_ = bottom[0]->count() / (N_ * outer_num_);
    // Gradient with respect to bottom data
    int nthreads = N_;
    FisherSumSigmaBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff_sigma, outer_num_, N_, inner_num_);
    nthreads = outer_num_ * N_ * inner_num_;
    FisherSumBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FisherSumLayer);

}  // namespace caffe
