import numpy as np
import tensorflow as tf


def time_to_batch(inputs, rate):
    '''If necessary zero-pads inputs and reshape by rate.
    
    Used to perform 1D dilated convolution.
    
    Args:
      inputs: (tensor) 
      rate: (int)
    Outputs:
      outputs: (tensor)
      pad_left: (int)
    '''
    batch_size = tf.shape(inputs)[0]
    width = tf.shape(inputs)[1]
    _, _, channels = inputs.get_shape().as_list()

    # Also add rate to width to make convolutional causal.
    width_pad = tf.to_int32(rate * (tf.ceil(tf.to_float(width) / rate) + 1))
    pad_left = width_pad - width

    zeros = tf.zeros(shape=(batch_size, pad_left, channels))
    padded = tf.concat(1, (zeros, inputs))
    padded_reshape = tf.reshape(padded, (batch_size * tf.to_int32((width_pad / rate)), rate,
                                         channels))
    outputs = tf.transpose(padded_reshape, perm=(1, 0, 2))
    return outputs, pad_left - rate


def batch_to_time(inputs, crop_left, rate):
    ''' Reshape to 1d signal, and remove excess zero-padding.
    
    Used to perform 1D dilated convolution.
    
    Args:
      inputs: (tensor)
      crop_left: (int)
      rate: (int)
    Ouputs:
      outputs: (tensor)
    '''
    batch_size = tf.to_int32(tf.shape(inputs)[0] / rate)
    width = tf.shape(inputs)[1]
    _, _, channels = inputs.get_shape().as_list()
    out_width = tf.to_int32(width * rate)

    inputs_transposed = tf.transpose(inputs, perm=(1, 0, 2))
    inputs_reshaped = tf.reshape(inputs_transposed,
                                 (batch_size, out_width, tf.to_int32(channels)))
    outputs = tf.slice(inputs_reshaped, [0, crop_left, 0], [-1, -1, -1])
    return outputs


def conv1d(inputs,
           out_channels,
           filter_width=2,
           stride=1,
           padding='VALID',
           data_format='NHWC',
           gain=np.sqrt(2),
           activation=tf.nn.relu,
           bias=False):
    '''One dimension convolution helper function.
    
    Sets variables with good defaults.
    
    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      data_format:
      gain:
      activation:
      bias:
      
    Outputs:
      outputs:
    '''
    in_channels = inputs.get_shape().as_list()[-1]

    stddev = gain / np.sqrt(filter_width**2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)

    w = tf.get_variable(name='w',
                        shape=(filter_width, in_channels, out_channels),
                        initializer=w_init)

    outputs = tf.nn.conv1d(inputs,
                           w,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)

    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name='b',
                            shape=(out_channels, ),
                            initializer=b_init)

        outputs = outputs + tf.expand_dims(tf.expand_dims(b, 0), 0)

    if activation:
        outputs = activation(outputs)

    return outputs


def dilated_conv1d(inputs,
                   out_channels,
                   filter_width=2,
                   rate=1,
                   padding='VALID',
                   name=None,
                   gain=np.sqrt(2),
                   activation=tf.nn.relu):
    '''
    
    Args:
      inputs: (tensor)
      output_channels:
      filter_width:
      rate:
      padding:
      name:
      gain:
      activation:

    Outputs:
      outputs: (tensor)
    '''
    assert name
    with tf.variable_scope(name):
        inputs_, pad_left = time_to_batch(inputs, rate=rate)
        outputs_ = conv1d(inputs_,
                          out_channels=out_channels,
                          filter_width=filter_width,
                          padding=padding,
                          gain=gain,
                          activation=activation)

        outputs = batch_to_time(outputs_, pad_left, rate=rate)

        # Add additional shape information.
        outputs.set_shape(tf.TensorShape([tf.Dimension(None), tf.Dimension(
            None), tf.Dimension(out_channels)]))

    return outputs


def _causal_linear(inputs, state, name=None, activation=None):
    assert name
    '''
    '''
    with tf.variable_scope(name, reuse=True) as scope:
        w = tf.get_variable('w')
        w_r = w[0, :, :]
        w_e = w[1, :, :]

        output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r)

        if activation:
            output = activation(output)
    return output


def _output_linear(h, name=''):
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('w')[0, :, :]
        b = tf.get_variable('b')

        output = tf.matmul(h, w) + tf.expand_dims(b, 0)
    return output


class Queue(object):
    def __init__(self, batch_size, state_size, buffer_size, name=None):
        assert name
        self.batch_size = batch_size
        self.state_size = state_size
        self.buffer_size = buffer_size

        with tf.variable_scope(name):
            self.state_buffer = tf.get_variable(
                'state_buffer',
                dtype=tf.float32,
                shape=[buffer_size, batch_size, state_size],
                initializer=tf.constant_initializer(0.0))

            self.pointer = tf.get_variable('pointer',
                                           initializer=tf.constant(0))

    def pop(self):
        state = tf.slice(self.state_buffer, [self.pointer, 0, 0],
                         [1, -1, -1])[0, :, :]
        return state

    def push(self, item):
        update_op = tf.scatter_update(self.state_buffer, self.pointer, item)
        with tf.control_dependencies([update_op]):
            push_op = tf.assign(self.pointer, tf.mod(self.pointer + 1,
                                                     self.buffer_size))
        return push_op
