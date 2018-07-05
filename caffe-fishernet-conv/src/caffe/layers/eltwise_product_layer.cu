#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void EltwiseProductForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* top_data,
          const Dtype* weight, const Dtype* bias,
          const int inner_num_, const int N_, const int K_,
          const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / (N_ * dim);
    const int n = (index % (N_ * dim)) / dim;
    const int k = (index % dim) / inner_num_;
    const int j = index % inner_num_;
    top_data[index] = weight[n * K_ + k] * bottom_data[i * dim + k * inner_num_ + j] 
      + bias[n * K_ + k];
  }
}

template <typename Dtype>
__global__ void EltwiseProduct1ForwardGPU(const int nthreads,
          Dtype* top_data, const Dtype* weight) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype epsilon = 0.0000001;
    top_data[index] = abs(weight[index]);
    if (top_data[index] < epsilon) {
      top_data[index] = epsilon;
    }
  }
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = this->blobs_[1]->gpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int inner_num_ = dim / K_;
  int nthreads = outer_num_ * N_ * dim;
  EltwiseProductForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, weight, bias, 
                                inner_num_, N_, K_, dim);
  if (top.size() >= 2) {
    nthreads = N_ * K_;
    Dtype* top_data1 = top[1]->mutable_gpu_data();
    EltwiseProduct1ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data1, weight);
  }
}

template <typename Dtype>
__global__ void EltwiseProductWeightBackwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* top_diff, Dtype* weight_diff, 
          const int outer_num_, const int inner_num_, const int N_, 
          const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int k = index / N_;
    const int n = index % N_;
    weight_diff[index] = 0;
    for (int i = 0; i < outer_num_; i++) {
      for (int j = 0; j < inner_num_; j++) {
        weight_diff[index] += bottom_data[i * dim + k * inner_num_ + j]
          * top_diff[i * N_ * dim + n * dim + k * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
__global__ void EltwiseProductWeight1BackwardGPU(const int nthreads,
          const Dtype* weight, const Dtype* top_diff, Dtype* weight_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    weight_diff[index] += top_diff[index] * (weight[index] > 0.0 ? 1.0 : -1.0);
  }
}

template <typename Dtype>
__global__ void EltwiseProductBiasBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bias_diff, 
          const int outer_num_, const int inner_num_, const int N_, 
          const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int k = index / N_;
    const int n = index % N_;
    bias_diff[index] = 0;
    for (int i = 0; i < outer_num_; i++) {
      for (int j = 0; j < inner_num_; j++) {
        bias_diff[index] += top_diff[i * N_ * dim + n * dim + k * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
__global__ void EltwiseProductBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, const Dtype* weight, 
          const int inner_num_, const int N_, const int K_,
          const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index / dim;
    const int k = (index % dim) / inner_num_;
    const int j = index % inner_num_;
    bottom_diff[index] = 0;
    for (int n = 0; n < N_; n++) {
      bottom_diff[index] += weight[n * K_ + k]
        * top_diff[i * N_ * dim + n * dim + k * inner_num_ + j];
    }
  }
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const int dim = bottom[0]->count() / outer_num_;
    const int inner_num_ = dim / K_;
    // Gradient with respect to weight
    int nthreads = N_ * K_;
    EltwiseProductWeightBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_diff, weight_diff, 
                                  outer_num_, inner_num_, N_, dim);
  }
  if (this->param_propagate_down_[0] && top.size() >= 2) {
    const Dtype* top_diff = top[1]->gpu_diff();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    int nthreads = N_ * K_;
    EltwiseProductWeight1BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, weight, top_diff, weight_diff);
  }

  if (this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    const int dim = bottom[0]->count() / outer_num_;
    const int inner_num_ = dim / K_;
    // Gradient with respect to bias
    int nthreads = N_ * K_;
    EltwiseProductBiasBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bias_diff, 
                                  outer_num_, inner_num_, N_, dim);
  }
   
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    const int dim = bottom[0]->count() / outer_num_;
    const int inner_num_ = dim / K_;
    // Gradient with respect to bottom data
    int nthreads = outer_num_ * dim;
    EltwiseProductBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, weight, 
                                  inner_num_, N_, K_, dim);

  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseProductLayer);

}  // namespace caffe
