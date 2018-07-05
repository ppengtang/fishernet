#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* top_data,
          const int N_, const int K_, const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (N_ * inner_num_);
    const int n = (index % (N_ * inner_num_)) / inner_num_;
    const int j = index % inner_num_;
    const int channels = N_ * K_;
    top_data[index] = 0;
    for (int k = 0; k < K_; k++) {
      top_data[index] += 
        bottom_data[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j];
    }
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int channels = bottom[0]->shape(1);
  const int K_ = channels / N_;
  const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
  const int nthreads = outer_num_ * N_ * inner_num_;
  SumForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, N_, K_, inner_num_);
}

template <typename Dtype>
__global__ void SumBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, 
          const int N_, const int K_, const int channels, 
          const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (channels * inner_num_);
    const int n = (index % (channels * inner_num_)) / (K_ * inner_num_);
    const int j = index % inner_num_;
    bottom_diff[index] = top_diff[i * N_ * inner_num_ + n * inner_num_ + j];
  }
}

template <typename Dtype>
void SumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int channels = bottom[0]->shape(1);
    const int K_ = channels / N_;
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    int nthreads = outer_num_ * N_ * K_ * inner_num_;
    SumBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, N_, K_, 
                                  channels, inner_num_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SumLayer);

}  // namespace caffe
