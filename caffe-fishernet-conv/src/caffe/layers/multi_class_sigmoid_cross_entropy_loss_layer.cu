#include "caffe/layers/multi_class_sigmoid_cross_entropy_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MulticlassSigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, Dtype* loss) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    loss[index] = log(1 + exp(-input_data[index]))
        + (1.0 - label[index]) * input_data[index];
  }
}

template <typename Dtype>
void MulticlassSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int nthreads = batch_size * channels;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  // NOLINT_NEXT_LINE(whitespace/operators)
  MulticlassSigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, loss_data);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= batch_size;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void MulticlassSigmoidCrossEntropyLossBackwardGPU(const int nthreads, 
		  const Dtype* input_data, const Dtype* label, Dtype* bottom_diff, 
                  const int channels) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = (1.0 / (1 + exp(-input_data[index])) 
        - label[index]) / channels;
  }
}

template <typename Dtype>
void MulticlassSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = batch_size * channels;
    
    MulticlassSigmoidCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, bottom_diff,
        channels);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    
    caffe_gpu_scal(bottom[0]->count(), loss_weight / batch_size, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MulticlassSigmoidCrossEntropyLossLayer);

}  // namespace caffe
