#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
void FisherScaleNewLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
      << "the bottom[0]->shape(1) must equal to bottom[1]->shape(1)";
  top[0]->ReshapeLike(*bottom[0]);
  outer_num_ = bottom[0]->shape(0);
  N_ = bottom[0]->shape(1);
}

template <typename Dtype>
void FisherScaleNewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_weight = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int inner_num_ = bottom[0]->count() / (N_ * outer_num_);
  for (int i = 0; i < outer_num_; i++) {
    for (int n = 0; n < N_; n++) {
      for (int j = 0; j < inner_num_; j++) {
        top_data[i * N_ * inner_num_ + n * inner_num_ + j] = bottom_data_weight[n]
          * bottom_data[i * N_ * inner_num_ + n * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
void FisherScaleNewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[1]->cpu_data();
    const int inner_num_ = bottom[0]->count() / (N_ * outer_num_);
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num_; i++) {
      for (int n = 0; n < N_; n++) {
        for (int j = 0; j < inner_num_; j++) {
          bottom_diff[i * N_ * inner_num_ + n * inner_num_ + j] = bottom_data[n]
            * top_diff[i * N_ * inner_num_ + n * inner_num_ + j];
        }
      }
    }
  }

  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int inner_num_ = bottom[0]->count() / (N_ * outer_num_);
    // Gradient with respect to bottom data
    for (int n = 0; n < N_; n++) {
      bottom_diff[n] = 0;
      for (int i = 0; i < outer_num_; i++) {
        for (int j = 0; j < inner_num_; j++) {
          bottom_diff[n] += bottom_data[i * N_ * inner_num_ + n * inner_num_ + j]
            * top_diff[i * N_ * inner_num_ + n * inner_num_ + j];
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FisherScaleNewLayer);
#endif

INSTANTIATE_CLASS(FisherScaleNewLayer);
REGISTER_LAYER_CLASS(FisherScaleNew);

}  // namespace caffe