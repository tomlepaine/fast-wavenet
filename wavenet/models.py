import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from layers import (Queue, _causal_linear, _output_linear, conv1d,
                    dilated_conv1d)


class Model(object):
    def __init__(self, num_time_samples, num_channels, gpu_fraction):
        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        h = inputs
        for b in range(2):
            for i in range(14):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, 128, rate=rate, name=name)

        outputs = conv1d(h,
                         256,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            outputs, targets))

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.initialize_all_variables())

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.cost = cost
        self.train_step = train_step
        self.sess = sess

    def _train(self, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        cost, _ = self.sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost

    def train(self, inputs, targets):
        losses = []
        terminal = False
        i = 0
        while not terminal:
            i += 1
            cost = self._train(inputs, targets)
            if cost < 1e-1:
                terminal = True
            losses.append(cost)
            if i % 50 == 0:
                plt.plot(losses)
                plt.show()


class Generator(object):
    def __init__(self, model, batch_size=1, input_size=1):
        self.model = model
        self.bins = np.linspace(-1, 1, 256)

        inputs = tf.placeholder(tf.float32, [batch_size, input_size],
                                name='inputs')

        print('Make Generator.')

        count = 0
        h = inputs

        push_ops = []
        for b in range(2):
            for i in range(14):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:
                    state_size = 1
                else:
                    state_size = 128

                q = Queue(batch_size=batch_size,
                          state_size=state_size,
                          buffer_size=rate,
                          name=name)

                state_ = q.pop()
                push = q.push(h)
                push_ops.append(push)
                h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                count += 1

        outputs = _output_linear(h)

        out_ops = [tf.argmax(tf.nn.softmax(outputs), 1)]
        out_ops.extend(push_ops)

        # Initialize new variables
        new_vars = [var for var in tf.trainable_variables()
                    if 'pointer' in var.name or 'state_buffer' in var.name]
        self.model.sess.run(tf.initialize_variables(new_vars))

        self.inputs = inputs
        self.out_ops = out_ops

    def run(self, input):

        predictions = []
        for step in range(32000):

            feed_dict = {self.inputs: input}
            outputs = self.model.sess.run(self.out_ops, feed_dict=feed_dict)
            output = outputs[0]  # ignore push ops

            input = self.bins[output][:, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_
