#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FisherScaleForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* bottom_data_gamma,
          Dtype* top_data, const int N_, const int K_, 
          const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (N_ * K_ * inner_num_);
    const int n = (index % (N_ * K_ * inner_num_)) / (K_ * inner_num_);
    const int j = index % inner_num_;
    top_data[index] = bottom_data[index] 
      * bottom_data_gamma[i * N_ * inner_num_ + n * inner_num_ + j];
  }
}

template <typename Dtype>
void FisherScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_data_gamma = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int channels = bottom[0]->shape(1);
  const int N_ = bottom[1]->shape(1);
  const int K_ = channels / N_;
  const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
  const int nthreads = outer_num_ * channels * inner_num_;
  FisherScaleForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, bottom_data_gamma, top_data, 
                                N_, K_, inner_num_);
}

template <typename Dtype>
__global__ void FisherScaleBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, 
          const Dtype* bottom_data, const int N_, 
          const int K_, const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (N_ * K_ * inner_num_);
    const int n = (index % (N_ * K_ * inner_num_)) / (K_ * inner_num_);
    const int j = index % inner_num_;
    bottom_diff[index] = bottom_data[i * N_ * inner_num_ + n * inner_num_ + j] 
      * top_diff[index];
  }
}

template <typename Dtype>
__global__ void FisherScaleGammaBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, 
          const Dtype * bottom_data, const int N_, 
          const int K_, const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (N_ * inner_num_ );
    const int n = (index % (N_ * inner_num_)) / inner_num_;
    const int j = index % inner_num_;
    const int tmp = i * N_ * K_ * inner_num_ + n * K_ * inner_num_;
    bottom_diff[index] = 0;
    for (int k = 0; k < K_; k++) {
      bottom_diff[index] += bottom_data[tmp + k * inner_num_ + j] 
        * top_diff[tmp + k * inner_num_ + j];
    }
  }
}

template <typename Dtype>
void FisherScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[1]->gpu_data();
    const int channels = bottom[0]->shape(1);
    const int N_ = bottom[1]->shape(1);
    const int K_ = channels / N_;
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    int nthreads = outer_num_ * channels * inner_num_;
    FisherScaleBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, bottom_data, N_, K_, inner_num_);
  }
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const int channels = bottom[0]->shape(1);
    const int N_ = bottom[1]->shape(1);
    const int K_ = channels / N_;
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    int nthreads = outer_num_ * N_ * inner_num_;
    FisherScaleGammaBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, bottom_data, N_, K_, inner_num_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FisherScaleLayer);

}  // namespace caffe
