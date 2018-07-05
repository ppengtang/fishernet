#include <vector>

#include "caffe/layers/l2_normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void L2NormalizeNormForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* norm_data,
          const int inner_num_, const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype epsilon = 1e-12;
    const int i = index / inner_num_;
    const int j = index % inner_num_;
    norm_data[index] = epsilon;
    for (int c = 0; c < channels; c++) {
      norm_data[index] += bottom_data[i * channels * inner_num_ + c * inner_num_ + j]
        * bottom_data[i * channels * inner_num_ + c * inner_num_ + j];
    }
    norm_data[index] = pow(norm_data[index], Dtype(0.5));
  }
}

template <typename Dtype>
__global__ void L2NormalizeForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* norm_data,
          Dtype* top_data, const int inner_num_, 
          const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (channels * inner_num_);
    const int j = index % inner_num_;
    top_data[index] = bottom_data[index] / norm_data[i * inner_num_ + j];
  }
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  const int channels = bottom[0]->shape(1);
  const int inner_num_ = bottom[0]->count() / (outer_num_ * channels);
  int nthreads = outer_num_ * inner_num_;
  L2NormalizeNormForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, norm_data, inner_num_, 
                                channels);
  nthreads = outer_num_ * channels * inner_num_;
  L2NormalizeForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, norm_data, top_data, inner_num_, 
                                channels);
}

template <typename Dtype>
__global__ void L2NormalizeBackwardGPU(const int nthreads,
          const Dtype* top_diff, const Dtype* top_data, 
          const Dtype* norm_data, Dtype* bottom_diff,
          const int inner_num_, const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (channels * inner_num_);
    const int j = index % inner_num_;
    bottom_diff[index] = top_diff[index] * (1 - top_data[index] * top_data[index])
      / norm_data[i * inner_num_ + j];
  }
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* norm_data = norm_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int channels = bottom[0]->shape(1);
  const int inner_num_ = bottom[0]->count() / (outer_num_ * channels);
  const int nthreads = outer_num_ * channels * inner_num_;
  L2NormalizeBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, top_data, norm_data, bottom_diff,
                                inner_num_, channels);
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormalizeLayer);
}  // namespace caffe
