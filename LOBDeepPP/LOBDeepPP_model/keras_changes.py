# Methods were taken from
# https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
# at 03.07.2019

import tensorflow as tf
from keras.backend.tensorflow_backend import dtype
from keras.backend.tensorflow_backend import StrictVersion
from keras.backend.tensorflow_backend import _has_nchw_support
from keras.backend.tensorflow_backend import normalize_data_format
from keras.backend.tensorflow_backend import _preprocess_conv1d_input
from keras.backend.tensorflow_backend import spatial_2d_padding
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _preprocess_padding
from keras.backend.tensorflow_backend import image_data_format
from keras.backend.tensorflow_backend import temporal_padding

import keras.backend.tensorflow_backend as K


# New function
def temporal_2d_padding(x, padding=(1, 1), data_format=None):
    """Pads the middle dimension of a 4D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A padded 4D tensor.
    """
    return spatial_2d_padding(
        x,
        padding=(padding, (0, 0)),
        data_format=data_format)


# New function
def temporal_3d_padding(x, padding=(1, 1), data_format=None):
    """Pads the middle dimension of a 5D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A padded 5D tensor.
    """
    return K.spatial_3d_padding(
        x,
        padding=(padding, (0, 0), (0, 0)),
        data_format=data_format)


# Added force_transpose as argument
def _preprocess_conv3d_input(x, data_format, force_transpose=False):
    """Transpose and cast the input before the conv3d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    # tensorflow doesn't support float64 for conv layer before 1.8.0
    if (dtype(x) == 'float64'
        and (StrictVersion(tf.__version__.split('-')[0]) < StrictVersion('1.8.0'))):
        x = tf.cast(x, 'float32')
    tf_data_format = 'NDHWC'
    if data_format == 'channels_first':
        # ADDED: option force_transpose in following line
        if not _has_nchw_support() or force_transpose:
            # ADDED till here
            x = tf.transpose(x, (0, 2, 3, 4, 1))
        else:
            tf_data_format = 'NCDHW'
    return x, tf_data_format


# Added option padding='causal'
def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 2 integers.
    # Returns
        A tensor, result of 2D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = K.normalize_data_format(data_format)

    # ADDED
    kernel_shape = kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv2d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        left_pad = dilation_rate[0] * (kernel_shape[0] - 1)
        x = temporal_2d_padding(x, (left_pad, 0), data_format=data_format)
        padding = 'valid'
    # ADDED till here

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)

    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


# Added option padding='causal'
def conv3d(x, kernel, strides=(1, 1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1, 1)):
    """3D convolution.
    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"`, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
            Whether to use Theano or TensorFlow/CNTK data format
            for inputs/kernels/outputs.
        dilation_rate: tuple of 3 integers.
    # Returns
        A tensor, result of 3D convolution.
    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)

    # ADDED
    kernel_shape = kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv3d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        left_pad = dilation_rate[0] * (kernel_shape[0] - 1)
        x = temporal_2d_padding(x, (left_pad, 0), data_format=data_format)
        padding = 'valid'
    # ADDED till here

    x, tf_data_format = _preprocess_conv3d_input(x, data_format)
    padding = _preprocess_padding(padding)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=dilation_rate,
        strides=strides,
        padding=padding,
        data_format=tf_data_format)
    if data_format == 'channels_first' and tf_data_format == 'NDHWC':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x


def separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1,
                     padding='valid', data_format=None, dilation_rate=1):
    """1D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: stride integer.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilation rate.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `data_format` is neither `channels_last` or `channels_first`.
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))

    # ADDED
    kernel_shape = depthwise_kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        # tmp_d = dilation_rate[0] if isinstance(dilation_rate, tuple) else
        # dilation_rate
        # left_pad = tmp_d * (kernel_shape[0] - 1)

        left_pad = dilation_rate[0] * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    # ADDED till here
    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        spatial_start_dim = 1
        strides = (1,) + strides * 2 + (1,)
    else:
        spatial_start_dim = 2
        strides = (1, 1) + strides * 2
    x = tf.expand_dims(x, spatial_start_dim)
    depthwise_kernel = tf.expand_dims(depthwise_kernel, 0)
    pointwise_kernel = tf.expand_dims(pointwise_kernel, 0)
    dilation_rate = (1,) + dilation_rate

    x = tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
                               strides=strides,
                               padding=padding,
                               rate=dilation_rate,
                               data_format=tf_data_format)

    x = tf.squeeze(x, [spatial_start_dim])

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW

    return x


K.temporal_2d_padding = temporal_2d_padding
K.temporal_3d_padding = temporal_3d_padding
K._preprocess_conv3d_input = _preprocess_conv3d_input
K.conv2d = conv2d
K.conv3d = conv3d
K.separable_conv1d = separable_conv1d
