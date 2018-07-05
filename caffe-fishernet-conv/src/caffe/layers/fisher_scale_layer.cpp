#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
void FisherScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "the number of bottom[0] must equal to bottom[1]";
  CHECK_EQ(bottom[0]->count(2), bottom[1]->count(2))
      << "the number of bottom[0]->count(2) must equal to bottom[0]->count(2)";
  top[0]->ReshapeLike(*bottom[0]);
  outer_num_ = bottom[0]->shape(0);
}

template <typename Dtype>
void FisherScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_gamma = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int channels = bottom[0]->shape(1);
  const int N_ = bottom[1]->shape(1);
  const int K_ = channels / N_;
  const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
  for (int i = 0; i < outer_num_; i++) {
  	for (int n = 0; n < N_; n++) {
  	  for (int k = 0; k < K_; k++) {
        for (int j = 0; j < inner_num_; j++) {
          top_data[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j] = 
            bottom_data[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j]
            * bottom_data_gamma[i * N_ * inner_num_ + n * inner_num_ + j];
        }
  	  }
  	}
  }
}

template <typename Dtype>
void FisherScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[1]->cpu_data();
    const int channels = bottom[0]->shape(1);
    const int N_ = bottom[1]->shape(1);
    const int K_ = channels / N_;
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num_; i++) {
      for (int n = 0; n < N_; n++) {
        for (int j = 0; j < inner_num_; j++) {
          for (int k = 0; k < K_; k++) {
            bottom_diff[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j] = 
              bottom_data[i * N_ * inner_num_ + n * inner_num_ + j]
              * top_diff[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j];
          }
        }
      }
    }
  }
  if (propagate_down[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int channels = bottom[0]->shape(1);
    const int N_ = bottom[1]->shape(1);
    const int K_ = channels / N_;
    const int inner_num_ = bottom[0]->count() / (channels * outer_num_);
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num_; i++) {
      for (int n = 0; n < N_; n++) {
        for (int j = 0; j < inner_num_; j++) {
          bottom_diff[i * N_ * inner_num_ + n * inner_num_ + j] = 0;
          for (int k = 0; k < K_; k++) {
            bottom_diff[i * N_ * inner_num_ + n * inner_num_ + j] += 
              bottom_data[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j]
              * top_diff[i * channels * inner_num_ + n * K_ * inner_num_ + k * inner_num_ + j];
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FisherScaleLayer);
#endif

INSTANTIATE_CLASS(FisherScaleLayer);
REGISTER_LAYER_CLASS(FisherScale);

}  // namespace caffe