from tensorflow.contrib.model_pruning.python import pruning
from TFRecords_encoder import create_tfrecords
from networks import Network
import tensorflow as tf
import sys
import os

TRAIN_SIZE = 50000
TEST_SIZE = 10000
INPUT_SIZE = 32
INPUT_CHANNEL = 3
NUM_CLASSES = 10

TFRECORDS_DIR = './TFrecords'

NUM_EPOCHS = 30
BATCH_SIZE = 500
LEARNING_RATE = 1e-2


def parse(example):
    features = {
        'img': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string)
    }
    parse_example = tf.io.parse_single_example(example, features)
    img = parse_example['img']
    label = parse_example['label']
    img = tf.decode_raw(img, tf.float32)
    img = tf.reshape(img, [INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL])
    label = tf.decode_raw(label, tf.int64)
    label = tf.reshape(label, [])
    return img, label


def load_tfrecords(name='train'):
    file = os.path.join(TFRECORDS_DIR, name + '.tfrecords')
    ds = tf.data.TFRecordDataset(file)
    ds = ds.map(parse)
    ds = ds.shuffle(1024)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.repeat()
    it = ds.make_one_shot_iterator()
    next_data = it.get_next()
    return next_data


def train_without_pruning():
    tf.compat.v1.reset_default_graph()

    # Inference
    network = Network(NUM_CLASSES)
    inputs = tf.compat.v1.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], 'inputs')
    logits = network.inference(inputs)

    # loss & accuracy
    labels = tf.compat.v1.placeholder(tf.int64, [None, ], 'labels')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    prediction = tf.argmax(tf.nn.softmax(logits), axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), dtype=tf.float32))

    # optimizer
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step)

    # loading data
    train_next = load_tfrecords('train')
    test_next = load_tfrecords('test')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # summaries
        logs_dir = './logs/without_pruning'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        tf.compat.v1.summary.scalar('monitor/loss', loss)
        tf.compat.v1.summary.scalar('monitor/acc', acc)
        merged_summary_op = tf.compat.v1.summary.merge_all()
        train_summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(logs_dir, 'train'), graph=sess.graph)
        test_summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(logs_dir, 'test'), graph=sess.graph)

        best_acc = 0
        saver = tf.compat.v1.train.Saver()
        for epoch in range(NUM_EPOCHS):
            # training
            num_steps = TRAIN_SIZE // BATCH_SIZE
            train_acc = 0
            train_loss = 0
            for step in range(num_steps):
                x, y = sess.run(train_next)
                _, summary, train_acc_batch, train_loss_batch = sess.run([train_op, merged_summary_op, acc, loss],
                                                                         feed_dict={inputs: x, labels: y})
                train_acc += train_acc_batch
                train_loss += train_loss_batch
                sys.stdout.write("\r epoch %d, step %d, training accuracy %g, training loss %g" %
                                 (epoch + 1, step + 1, train_acc_batch, train_loss_batch))
                sys.stdout.flush()
                train_summary_writer.add_summary(summary, global_step=epoch * num_steps + step)
                train_summary_writer.flush()
            print("\n epoch %d, training accuracy %g, training loss %g" %
                  (epoch + 1, train_acc / num_steps, train_loss / num_steps))

            # testing
            num_steps = TEST_SIZE // BATCH_SIZE
            test_acc = 0
            test_loss = 0
            for step in range(num_steps):
                x, y = sess.run(test_next)
                summary, test_acc_batch, test_loss_batch = sess.run([merged_summary_op, acc, loss],
                                                                    feed_dict={inputs: x, labels: y})
                test_acc += test_acc_batch
                test_loss += test_loss_batch
                test_summary_writer.add_summary(summary, global_step=(epoch * num_steps + step) * (TRAIN_SIZE // TEST_SIZE))
                test_summary_writer.flush()
            print(" epoch %d, testing accuracy %g, testing loss %g" %
                  (epoch + 1, test_acc / num_steps, test_loss / num_steps))

            if test_acc / num_steps > best_acc:
                best_acc = test_acc / num_steps
                saver.save(sess, './ckpt_without_pruning/model')

        print(" Best Testing Accuracy %g" % best_acc)


def train_with_pruning():
    tf.compat.v1.reset_default_graph()

    # Inference
    network = Network(NUM_CLASSES)
    inputs = tf.compat.v1.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL], 'inputs')
    logits = network.pruning_inference(inputs)

    # loss & accuracy
    labels = tf.compat.v1.placeholder(tf.int64, [None, ], 'labels')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    prediction = tf.argmax(tf.nn.softmax(logits), axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), dtype=tf.float32))

    # Create pruning operator
    global_step = tf.train.get_or_create_global_step()
    pruning_hparams = pruning.get_pruning_hparams()
    pruning_hparams.sparsity_function_end_step = 1000
    p = pruning.Pruning(pruning_hparams, global_step=global_step)
    mask_update_op = p.conditional_mask_update_op()
    p.add_pruning_summaries()

    # optimizer
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step)

    # loading data
    train_next = load_tfrecords('train')
    test_next = load_tfrecords('test')

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # summaries
        logs_dir = './logs/with_pruning'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        tf.compat.v1.summary.scalar('monitor/loss', loss)
        tf.compat.v1.summary.scalar('monitor/acc', acc)
        merged_summary_op = tf.compat.v1.summary.merge_all()
        train_summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(logs_dir, 'train'), graph=sess.graph)
        test_summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(logs_dir, 'test'), graph=sess.graph)

        best_acc = 0
        saver = tf.compat.v1.train.Saver()
        for epoch in range(NUM_EPOCHS):
            # training
            num_steps = TRAIN_SIZE // BATCH_SIZE
            train_acc = 0
            train_loss = 0
            for step in range(num_steps):
                x, y = sess.run(train_next)
                _, summary, train_acc_batch, train_loss_batch = sess.run([train_op, merged_summary_op, acc, loss],
                                                                         feed_dict={inputs: x, labels: y})
                sess.run(mask_update_op)
                train_acc += train_acc_batch
                train_loss += train_loss_batch
                sys.stdout.write("\r epoch %d, step %d, training accuracy %g, training loss %g" %
                                 (epoch + 1, step + 1, train_acc_batch, train_loss_batch))
                sys.stdout.flush()
                train_summary_writer.add_summary(summary, global_step=epoch * num_steps + step)
                train_summary_writer.flush()
            print("\n epoch %d, training accuracy %g, training loss %g" %
                  (epoch + 1, train_acc / num_steps, train_loss / num_steps))

            # testing
            num_steps = TEST_SIZE // BATCH_SIZE
            test_acc = 0
            test_loss = 0
            for step in range(num_steps):
                x, y = sess.run(test_next)
                summary, test_acc_batch, test_loss_batch = sess.run([merged_summary_op, acc, loss],
                                                                    feed_dict={inputs: x, labels: y})
                test_acc += test_acc_batch
                test_loss += test_loss_batch
                test_summary_writer.add_summary(summary, global_step=(epoch * num_steps + step) * (TRAIN_SIZE // TEST_SIZE))
                test_summary_writer.flush()
            print(" epoch %d, testing accuracy %g, testing loss %g" %
                  (epoch + 1, test_acc / num_steps, test_loss / num_steps))

            if test_acc / num_steps > best_acc:
                best_acc = test_acc / num_steps
                saver.save(sess, './ckpt_with_pruning/model')

        print(" Best Testing Accuracy %g" % best_acc)


if __name__ == '__main__':
    if os.path.exists(os.path.join(TFRECORDS_DIR, 'train.tfrecords')) and\
       os.path.exists(os.path.join(TFRECORDS_DIR, 'test.tfrecords')):
        print('TFRecords already exists!')
    else:
        create_tfrecords('train', TFRECORDS_DIR)
        create_tfrecords('test', TFRECORDS_DIR)
        print('TFRecords creating completed!')

    train_without_pruning()
    train_with_pruning()
