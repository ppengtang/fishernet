#include "caffe/layers/sum_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SumPoolingForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* top_data, const int channels,
          const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / channels;
    const int c = index % channels;
    top_data[index] = 0;
    for (int j = 0; j < inner_num_; j++) {
      top_data[i * channels + c] += bottom_data[i * channels * inner_num_ + c * inner_num_ + j];
    }
  }
}

template <typename Dtype>
void SumPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int channels = bottom[0]->shape(1);
  const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
  const int nthreads = outer_num_ * channels;
  SumPoolingForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, 
                                channels, inner_num_);
}

template <typename Dtype>
__global__ void SumPoolingBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, 
          const int channels, const int inner_num_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (channels * inner_num_);
    const int c = (index % (channels * inner_num_)) / inner_num_;
    bottom_diff[index] = top_diff[i * channels + c];
  }
}

template <typename Dtype>
void SumPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int channels = bottom[0]->shape(1);
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    const int nthreads = outer_num_ * channels * inner_num_;
    SumPoolingBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, channels, inner_num_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SumPoolingLayer);

}  // namespace caffe
