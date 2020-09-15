import tensorflow as tf
import numpy as np
import pickle
import time
import os


DATASET_DIR = './CIFAR-10'


with tf.Session() as sess:
    with tf.gfile.FastGFile('./pb_models/model_without_pruning.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    graph = tf.get_default_graph()
    inputs = graph.get_tensor_by_name('inputs:0')
    logits = graph.get_tensor_by_name('logits:0')

    with open(os.path.join(DATASET_DIR, 'test_batch'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        x = data[b'data']
        y = data[b'labels']
        correct = 0
        timer = time.time()
        for j in range(10000):

            img = np.reshape(x[j], [3, 32, 32])
            img = np.transpose(img, [1, 2, 0]).astype(np.uint8)
            img = np.expand_dims(img, 0)
            label = y[j]

            output = sess.run([logits], feed_dict={'inputs:0': img})

            if np.argmax(output) == label:
                correct += 1
            print(correct / (j + 1))

        print(time.time() - timer)
