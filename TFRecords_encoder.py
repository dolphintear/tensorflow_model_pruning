import tensorflow as tf
import numpy as np
import pickle
import os

DATASET_DIR = './CIFAR-10'
TFRECORDS_DIR = './TFrecords'


def create_tfrecords(name, tfrecords_dir):

    if not os.path.exists(DATASET_DIR):
        print('Invalid Database Path')
        return
    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    writer = tf.io.TFRecordWriter(os.path.join(tfrecords_dir, name + '.tfrecords'))
    if name == 'train':
        for i in range(1, 6):
            with open(os.path.join(DATASET_DIR, 'data_batch_' + str(i)), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                x = data[b'data']
                y = data[b'labels']
                for j in range(10000):
                    img = np.reshape(x[j], [3, 32, 32])
                    img = np.transpose(img, [1, 2, 0]).astype('float32') / 255.
                    label = (np.ones(1, dtype=np.int64) * y[j])

                    img = img.tobytes()
                    label = label.tobytes()

                    example = tf.train.Example(features=tf.train.Features(feature={
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
                    }))
                    writer.write(example.SerializeToString())

    elif name == 'test':
        with open(os.path.join(DATASET_DIR, 'test_batch'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            x = data[b'data']
            y = data[b'labels']
            for j in range(10000):
                img = np.reshape(x[j], [3, 32, 32])
                img = np.transpose(img, [1, 2, 0]).astype('float32') / 255.
                label = (np.ones(1, dtype=np.int64) * y[j])
                
                img = img.tobytes()
                label = label.tobytes()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
                }))
                writer.write(example.SerializeToString())

    else:
        return

    writer.close()


if __name__ == '__main__':
    create_tfrecords('train', TFRECORDS_DIR)
    create_tfrecords('test', TFRECORDS_DIR)
