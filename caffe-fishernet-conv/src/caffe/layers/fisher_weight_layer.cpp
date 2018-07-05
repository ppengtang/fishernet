#include "caffe/layers/fisher_layer.hpp"

namespace caffe {

template <typename Dtype>
void FisherWeightLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_component = this->layer_param_.fisher_weight_param().num_component();
  N_ = num_component;
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(1, N_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.fisher_weight_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void FisherWeightLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(2);
  top_shape[0] = 1;
  top_shape[1] = N_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void FisherWeightLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int n = 0; n < N_; n++) {
    top_data[n] = weight[n];
  }
}

template <typename Dtype>
void FisherWeightLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int n = 0; n < N_; n++) {
      weight_diff[n] = top_diff[n];
    }
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < bottom[0]->count(); i++) {
      bottom_diff[i] = 0;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FisherWeightLayer);
#endif

INSTANTIATE_CLASS(FisherWeightLayer);
REGISTER_LAYER_CLASS(FisherWeight);

}  // namespace caffe
