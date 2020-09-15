# tensorflow_model_pruning
Train a pruned CNN on CIFAR-10 dataset with TensorFlow model_pruning module

### Environment

- python==3.6.5
- tensorflow==1.14.0

### Getting Started

1. Download and extract the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html ) dataset in the "CIFAR-10" folder.
2. Run "TFRecords_encoder.py" to create TFRecords of the dataset.
3. Run "train.py" to train the network with and without pruning.
4. Run "frozen_graph.py" and "model_quantization.py" to convert model format.
5. Run "test_pb.py" and "test_tflite.py" to test models on CIFAR-10 test set.
