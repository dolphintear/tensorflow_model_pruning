import tensorflow as tf
import os


def pb2tflite(graph_def_file, tflite_file):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    input_names = ['inputs']
    output_names = ['logits']
    input_tensor = {input_names[0]: [1, 32, 32, 3]}
    
    # uint8 quantization
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()

    if not os.path.exists(os.path.dirname(tflite_file)):
        os.makedirs(os.path.dirname(tflite_file))
    open(tflite_file, 'wb').write(tflite_quant_model)


if __name__ == '__main__':
    pb2tflite('./pb_models/model_without_pruning.pb', './tflite_models/model_without_pruning.tflite')
    pb2tflite('./pb_models/model_with_pruning.pb', './tflite_models/model_with_pruning.tflite')
