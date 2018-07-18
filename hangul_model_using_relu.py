#!/usr/bin/env python

import argparse
import io
import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  './labels/2350-common-hangul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'saved-model')

MODEL_NAME = 'hangul_tensorflow'
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

DEFAULT_NUM_TRAIN_STEPS = 50000
BATCH_SIZE = 100


def get_image(files, num_classes):

    file_queue = tf.train.string_input_producer(files)

    reader = tf.TFRecordReader()

    key, example = reader.read(file_queue)

    features = tf.parse_single_example(
        example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })

    label = features['image/class/label']
    image_encoded = features['image/encoded']

    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH*IMAGE_HEIGHT])

    label = tf.stack(tf.one_hot(label, num_classes))
    return label, image


def export_model(model_output_dir, input_node_names, output_node_name):

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '.pb')
    with tf.gfile.FastGFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=5e-2)
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')


def main(label_file, tfrecords_dir, model_output_dir, num_train_steps):

    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_classes = len(labels)

    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.gfile.Glob(tf_record_pattern)
    label, image = get_image(train_data_files, num_classes)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)
    tlabel, timage = get_image(test_data_files, num_classes)

    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE,
        capacity=2000,
        min_after_dequeue=1000)

    timage_batch, tlabel_batch = tf.train.batch(
        [timage, tlabel], batch_size=BATCH_SIZE,
        capacity=2000)

    # Create Model
    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH*IMAGE_HEIGHT],
                       name=input_node_name)

    y_ = tf.placeholder(tf.float32, [None, num_classes])

    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # 첫번째 convolutional layer
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 두번째 convolutional layer 
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    # 두번째 pooling layer.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 세번째 convolutional layer
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 2, 2, 1], padding='SAME') + b_conv3)

    # 네번째 convolutional layer
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

    # 다섯번째 convolutional layer
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

    # Fully connected layer.
    h_pool_flat = tf.reshape(h_conv5, [-1, 4*4*128])
    W_fc1 = weight_variable([4*4*128, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout layer. 
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Classification layer.
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    

    tf.nn.softmax(y, name=output_node_name)

    # loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(y_),
            logits=y
        )
    )

    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        checkpoint_file = os.path.join(model_output_dir, MODEL_NAME + '.chkp')
        tf.train.write_graph(sess.graph_def, model_output_dir,
                             MODEL_NAME + '.pbtxt', True)

        for step in range(num_train_steps):

            train_images, train_labels = sess.run([image_batch, label_batch])

            sess.run(train_step, feed_dict={x: train_images, y_: train_labels,
                                            keep_prob: 0.5})

            if step % 100 == 0:
                train_accuracy = sess.run(
                    accuracy,
                    feed_dict={x: train_images, y_: train_labels,
                               keep_prob: 1.0}
                )
                print("Step %d, Training Accuracy %g" %
                      (step, float(train_accuracy)))

            if step % 10000 == 0:
                saver.save(sess, checkpoint_file, global_step=step)

        saver.save(sess, checkpoint_file)

        sample_count = 0
        for f in test_data_files:
            sample_count += sum(1 for _ in tf.python_io.tf_record_iterator(f))

        print('Testing model...')

        num_batches = int(sample_count/BATCH_SIZE) or 1
        total_correct_preds = 0

        accuracy2 = tf.reduce_sum(correct_prediction)
        for step in range(num_batches):
            image_batch2, label_batch2 = sess.run([timage_batch, tlabel_batch])
            acc = sess.run(accuracy2, feed_dict={x: image_batch2,
                                                 y_: label_batch2,
                                                 keep_prob: 1.0})
            total_correct_preds += acc

        accuracy_percent = total_correct_preds/(num_batches*BATCH_SIZE)
        print("Testing Accuracy {}".format(accuracy_percent))

        export_model(model_output_dir, [input_node_name, keep_prob_node_name],
                     output_node_name)

        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store saved model files.')
    parser.add_argument('--num-train-steps', type=int, dest='num_train_steps',
                        default=DEFAULT_NUM_TRAIN_STEPS,
                        help='Number of training steps to perform. This value '
                             'should be increased with more data. The number '
                             'of steps should cover several iterations over '
                             'all of the training data (epochs). Example: If '
                             'you have 15000 images in the training set, one '
                             'epoch would be 15000/100 = 150 steps where 100 '
                             'is the batch size. So, for 10 epochs, you would '
                             'put 150*10 = 1500 steps.')
    args = parser.parse_args()
    main(args.label_file, args.tfrecords_dir,
         args.output_dir, args.num_train_steps)
