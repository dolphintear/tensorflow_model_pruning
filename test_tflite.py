import tensorflow as tf
import numpy as np
import pickle
import time
import os


DATASET_DIR = './CIFAR-10'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='./tflite_models/model_without_pruning.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(os.path.join(DATASET_DIR, 'test_batch'), 'rb') as f:
    data = pickle.load(f, encoding='bytes')
    x = data[b'data']
    y = data[b'labels']
    correct = 0
    timer = time.time()
    for j in range(10000):

        img = np.reshape(x[j], [3, 32, 32])
        img = np.transpose(img, [1, 2, 0]).astype(np.float32) / 255.
        img = np.expand_dims(img, 0)
        label = y[j]

        interpreter.set_tensor(input_details[0]['index'], img)

        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if np.argmax(output) == label:
            correct += 1
        print(correct / (j + 1))

    print(time.time() - timer)
