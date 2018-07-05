#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_component = this->layer_param_.eltwise_product_param().num_component();
  N_ = num_component;
  K_ = bottom[0]->shape(1);
  outer_num_ = bottom[0]->shape(0);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    // Initialize the weights
    vector<int> weight_shape(1, N_ * K_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.eltwise_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // Intiialize and fill the bias term
    vector<int> bias_shape(1, N_ * K_);
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.eltwise_product_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = N_ * K_;
  top[0]->Reshape(top_shape);
  if (top.size() >= 2) {
    vector<int> top_shape1(2);
    top_shape1[0] = 1;
    top_shape1[1] = N_ * K_;
    top[1]->Reshape(top_shape1);
  }
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int inner_num_ = dim / K_;
  for (int i = 0; i < outer_num_; i++) {
    for (int n = 0; n < N_; n++) {
      for (int k = 0; k < K_; k++) {
        for (int j = 0; j < inner_num_; j++) {
          top_data[i * N_ * dim + n * dim + k * inner_num_ + j] = weight[n * K_ + k]
            * bottom_data[i * dim + k * inner_num_ + j] + bias[n * K_ + k];
        }
      }
    }
  }
  if (top.size() >= 2) {
    Dtype* top_data1 = top[1]->mutable_cpu_data();
    Dtype epsilon = 0.0000001;
    for (int n = 0; n < N_; n++) {
      for (int k = 0; k < K_; k++) {
        top_data1[n * K_ + k] = abs(weight[n * K_ + k]);
        if (top_data1[n * K_ + k] < epsilon) {
          top_data1[n * K_ + k] = epsilon;
        }
      }
    }
  }
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    const int dim = bottom[0]->count() / outer_num_;
    const int inner_num_ = dim / K_;
    // Gradient with respect to weight
    for (int n = 0; n < N_; n++) {
      for (int k = 0; k < K_; k++) {
      	weight_diff[n * K_ + k] = 0;
      	for (int i = 0; i < outer_num_; i++) {
          for (int j = 0; j < inner_num_; j++) {
            weight_diff[n * K_ + k] += bottom_data[i * dim + k * inner_num_ + j]
              * top_diff[i * N_ * dim + n * dim + k * inner_num_ + j];
          }
      	}
      }
    }
    if (top.size() >= 2) {
      const Dtype* top_diff1 = top[1]->cpu_diff();
      const Dtype* weight = this->blobs_[0]->cpu_data();
      for (int n = 0; n < N_; n++) {
        for (int k = 0; k < K_; k++) {
          weight_diff[n * K_ + k] += top_diff1[n * K_ + k] 
            * (weight[n * K_ + k] > 0.0 ? 1.0 : -1.0);
        }
      }
    }
  }
  if (this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    const int dim = bottom[0]->count() / outer_num_;
    const int inner_num_ = dim / K_;
    // Gradient with respect to bias
    for (int n = 0; n < N_; n++) {
      for (int k = 0; k < K_; k++) {
        bias_diff[n * K_ + k] = 0;
        for (int i = 0; i < outer_num_; i++) {
          for (int j = 0; j < inner_num_; j++) {
            bias_diff[n * K_ + k] += top_diff[i * N_ * dim + n * dim + k * inner_num_ + j];
          }
        }
      }
    }
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const int dim = bottom[0]->count() / outer_num_;
    const int inner_num_ = dim / K_;
    // Gradient with respect to bottom data
    for (int i = 0; i < outer_num_; i++) {
      for (int k = 0; k < K_; k++) {
        for (int j = 0; j < inner_num_; j ++) {
          bottom_diff[i * dim + k * inner_num_ + j] = 0;
          for (int n = 0; n < N_; n++) {
            bottom_diff[i * dim + k * inner_num_ + j] += weight[n * K_ + k]
              * top_diff[i * N_ * dim + n * dim + k * inner_num_ + j];
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseProductLayer);
#endif

INSTANTIATE_CLASS(EltwiseProductLayer);
REGISTER_LAYER_CLASS(EltwiseProduct);

}  // namespace caffe
