#include "caffe/layers/sum_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void SumPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2);
  outer_num_ = bottom[0]->shape(0);
  top_shape[0] = outer_num_;
  top_shape[1] = bottom[0]->shape(1);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SumPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int channels = bottom[0]->shape(1);
  const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
  for (int i = 0; i < outer_num_; i++) {
    for (int c = 0; c < channels; c++) {
      top_data[i * channels + c] = 0;
      for (int j = 0; j < inner_num_; j++) {
        top_data[i * channels + c] += bottom_data[i * channels * inner_num_ + c * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
void SumPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int channels = bottom[0]->shape(1);
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num_; i++) {
      for (int c = 0; c < channels; c++) {
        for (int j = 0; j < inner_num_; j++) {
          bottom_diff[i * channels * inner_num_ + c * inner_num_ + j] = top_diff[i * channels + c];
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SumPoolingLayer);
#endif

INSTANTIATE_CLASS(SumPoolingLayer);
REGISTER_LAYER_CLASS(SumPooling);

}  // namespace caffe