import tensorflow as tf
import json
import numpy as np
import data.loader as loader
from data.test_output import generate_my_json


class CNN:
    def __init__(self, xhc_size, lr=1e-4, max_input_length=250):
        with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):

            self.max_input_length = max_input_length
            self.xhc_size = xhc_size

            self.input = tf.placeholder(dtype=tf.float32, shape=[max_input_length, xhc_size[0]], name='sentence')
            self.label = tf.placeholder(dtype=tf.float32, shape=[2], name='label')

            self.out = self.Cond_conv(self.input)

            # is_positive = tf.cast(tf.equal(self.label, 1), tf.float32)
            # weight = 10. * is_positive + 1. * (1. - is_positive)
            self.loss = - tf.reduce_sum(tf.multiply(self.label, tf.log(self.out)))

            print("Building optimization")
            self.optm = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

            self.base_address = 'D:\\UserData\\DeepLearning\\CNN-for-Triples-Extraction\\'
            self.saver = tf.train.Saver(max_to_keep=3)

    def Cond_conv(self, input, name='Conditional_layer_CNN'):
        with tf.variable_scope(name):
            #  Input shape : [1, 250, 2, 1]
            input = input[np.newaxis, :, :, np.newaxis]
            output0 = self.set_conv(input, 1, 128, 'conv_layer0')    # Output0 shape : [1, 250, 2, 128]
            output1 = self.set_conv(output0, 5, 256, 'conv_layer1')  # Output1 shape : [1, 50, 2, 256]
            output2 = self.set_conv(output1, 2, 64, 'conv_layer2')  # Output2 shape : [1, 25, 2, 64]
            output3 = self.set_conv(output2, 5, 32, 'conv_layer3')  # Output3 shape : [1, 5, 2, 32]
            output3 = tf.reshape(output3, [1, 320])
            W = tf.get_variable('CNN_W', shape=[320, 2], dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
            b = tf.get_variable('CNN_b', shape=[2], dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
            condition = tf.nn.softmax(tf.squeeze(tf.add(tf.matmul(output3, W), b)))
            self.output_process = tf.squeeze(tf.add(tf.matmul(output3, W), b))
            return condition

    def set_conv(self, input, scale, channel, name='conv'):
        with tf.variable_scope(name):
            filter = tf.get_variable('filter', shape=[5, self.xhc_size[0], np.shape(input)[-1], channel], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
            b = tf.get_variable('b', shape=[channel], dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
            output = tf.nn.conv2d(input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.contrib.layers.batch_norm(output, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
            output = tf.nn.leaky_relu(tf.add(output, b))
            output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, scale, 1, 1], padding='SAME')
            return output

    def train(self, data_path, maxepoch, continue_train=False, trained_steps=0):
        data = json.load(open(data_path, 'r'))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if continue_train:
                latest = tf.train.latest_checkpoint(self.base_address + 'parameters/')
                self.saver.restore(sess, latest)

            writer = tf.summary.FileWriter(".\\Visual\\")
            writer.add_graph(sess.graph)
            #  Epoch
            for j in range(maxepoch):
                #  Sample in a Epoch
                for i in range(np.shape(data)[0]):
                    # i = 0
                    indexes = data[i]['indexes']
                    times = data[i]['times']
                    attributes = data[i]['attributes']
                    values = data[i]['values']
                    results = data[i]['results']
                    ddata, label, mask = loader.data_process(indexes, times, attributes, values, results,
                                                             self.max_input_length)

                    loss_array = []
                    out_array = []
                    A = len(results)
                    A_ = 0
                    A_and_A_ = 0
                    correct = 0
                    #  Each Sample has many combinations to input
                    for k in range(np.shape(ddata)[0]):
                        loss, out, _, output_process = sess.run([self.loss, self.out, self.optm, self.output_process],
                                                                 feed_dict={self.input: ddata[k],
                                                                            self.label: [label[k], 1-label[k]]})

                        loss_array.append(loss)
                        out_array.append(out)
                        if out[0] > 0.5:
                            A_ += 1
                            if label[k] > 0.5:
                                A_and_A_ += 1
                        if (out[0] - 0.5) * (label[k] - 0.5) > 0:
                            correct += 1.
                    accuracy = correct / np.shape(ddata)[0]
                    p = (A_and_A_ / A_) if A_ > 1e-5 else 0.
                    r = (A_and_A_ / A) if A > 1e-5 else 0.
                    F = (2 * p * r / (p + r)) if (p + r) > 1e-5 else 0.
                    var = np.var(out_array)
                    print('Epoch:%d  Sample:%d  Mean Loss:%05f' % (j, i, np.average(loss_array)), end=' ')
                    print('Accuracy: %05f Precise: %05f, Recall: %05f, F1 Score: %05f, Var: %f' % (accuracy, p, r, F, var), end=' ')
                    print('All: %d, A_: %d, Result: %d, Correct: %d, ALL=A_: %d' % (np.shape(ddata)[0], A_, A, correct, np.shape(ddata)[0]==A_))
                self.saver.save(sess, 'parameters/BLSTM', global_step=trained_steps+j)

    def test(self, data_path):
        data = json.load(open(data_path, 'r'))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            latest = tf.train.latest_checkpoint(self.base_address + 'parameters/')
            self.saver.restore(sess, latest)

            writer = generate_my_json()
            #  Sample in a Epoch
            for i in range(np.shape(data)[0]):
                indexes = data[i]['indexes']
                times = data[i]['times']
                attributes = data[i]['attributes']
                values = data[i]['values']
                ddata, label, mask = loader.data_process(indexes, times, attributes, values, max_input_length=self.max_input_length)
                #  Each Sample has many combinations to input
                for k in range(np.shape(ddata)[0]):
                    hout = sess.run(self.out, feed_dict={self.input: ddata[k]})
                    if hout[0] > 0.5:
                        writer.input_the_data(ddata[k].tolist(), 0)
                writer.input_the_data(ddata[0].tolist(), 1)
            writer.input_the_data([1, 2, 3], 1, 1)
