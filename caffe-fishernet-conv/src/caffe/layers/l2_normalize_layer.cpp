#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/l2_normalize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(top[0] != bottom[0]) << "do not support in place operation";
  outer_num_ = bottom[0]->shape(0);
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = 1;
  norm_.Reshape(top_shape);
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  const int channels = bottom[0]->shape(1);
  const int inner_num_ = bottom[0]->count() / (outer_num_ * channels);
  const Dtype epsilon = 1e-12;
  for (int i = 0; i < outer_num_; i++) {
    for (int j = 0; j < inner_num_; j++) {
      norm_data[i * inner_num_ + j] = epsilon;
      for (int c = 0; c < channels; c++) {
        norm_data[i * inner_num_ + j] += bottom_data[i * channels * inner_num_ + c * inner_num_ + j]
          * bottom_data[i * channels * inner_num_ + c * inner_num_ + j];
      }
      norm_data[i * inner_num_ + j] = pow(norm_data[i * inner_num_ + j], 0.5);
      for (int c = 0; c < channels; c++) {
        top_data[i * channels * inner_num_ + c * inner_num_ + j] = 
          bottom_data[i * channels * inner_num_ + c * inner_num_ + j] / norm_data[i * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
void L2NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int channels = bottom[0]->shape(1);
  const int inner_num_ = bottom[0]->count() / (outer_num_ * channels);
  for (int i = 0; i < outer_num_; i++) {
    for (int c = 0; c < channels; c++) {
      for (int j = 0; j < inner_num_; j++) {
        bottom_diff[i * channels * inner_num_ + c * inner_num_ + j] = 
          top_diff[i * channels * inner_num_ + c * inner_num_ + j] 
          * (1 - top_data[i * channels * inner_num_ + c * inner_num_ + j]
            * top_data[i * channels * inner_num_ + c * inner_num_ + j])
          / norm_data[i * inner_num_ + j];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(L2NormalizeLayer);
#endif

INSTANTIATE_CLASS(L2NormalizeLayer);
REGISTER_LAYER_CLASS(L2Normalize);

}  // namespace caffe
