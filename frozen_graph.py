from tensorflow.contrib.model_pruning.python import strip_pruning_vars_lib
from tensorflow.python.framework import graph_io
import tensorflow as tf
import tempfile
import zipfile
import os


def strip_pruning_vars(input_checkpoint, output_node_names, output_dir, filename):
    tf.compat.v1.reset_default_graph()
    output_node_names = output_node_names.replace(' ', '').split(',')

    initial_graph_def = strip_pruning_vars_lib.graph_def_from_checkpoint(
        input_checkpoint, output_node_names)

    for node in initial_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    final_graph_def = strip_pruning_vars_lib.strip_pruning_vars_fn(
        initial_graph_def, output_node_names)
    graph_io.write_graph(final_graph_def, output_dir, filename, as_text=False)


def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)


if __name__ == '__main__':
    strip_pruning_vars('./ckpt_without_pruning', 'logits', './pb_models', 'model_without_pruning.pb')
    strip_pruning_vars('./ckpt_with_pruning', 'logits', './pb_models', 'model_with_pruning.pb')
    print('Size of gzipped mode without pruning: %.2fMB' % (get_gzipped_model_size('./pb_models/model_without_pruning.pb') / 1024 / 1024))
    print('Size of gzipped mode with pruning: %.2fMB' % (get_gzipped_model_size('./pb_models/model_with_pruning.pb') / 1024 / 1024))
