from keras.backend.tensorflow_backend import normalize_data_format, \
    _preprocess_conv1d_input, temporal_padding, _preprocess_padding, \
    _preprocess_conv2d_input, spatial_2d_padding
import tensorflow as tf
# import keras.backend.tensorflow_backend as K


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
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation_rate, int) and False:
        dilation_rate = (dilation_rate,)
    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    if tf_data_format == 'NWC':
        tf_data_format = 'NHWC'
    else:
        tf_data_format = 'NCHW'
    # ADDED
    kernel_shape = depthwise_kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        tmp_d = dilation_rate[0]\
            if isinstance(dilation_rate, tuple) \
            else dilation_rate
        left_pad = tmp_d * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    # ADDED till here
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


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    """2D convolution with separable filters.

    # Arguments
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        pointwise_kernel: kernel for the 1x1 convolution.
        strides: strides tuple (length 2).
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: tuple of integers,
            dilation rates for the separable convolution.

    # Returns
        Output tensor.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)
    # ADDED
    kernel_shape = depthwise_kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    # ADDED till here
    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
    else:
        strides = (1, 1) + strides

    x = tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel,
                               strides=strides,
                               padding=padding,
                               rate=dilation_rate,
                               data_format=tf_data_format)
    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
    return x


def conv1d(x, kernel, strides=1, padding='valid',
           data_format=None, dilation_rate=1):
    """1D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: stride integer.
        padding: string, `"same"`, `"causal"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        dilation_rate: integer dilate rate.

    # Returns
        A tensor, result of 1D convolution.

    # Raises
        ValueError: If `data_format` is neither
            `"channels_last"` nor `"channels_first"`.
    """
    data_format = normalize_data_format(data_format)
    kernel_shape = kernel.get_shape().as_list()
    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        # causal (dilated) convolution:
        left_pad = dilation_rate * (kernel_shape[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    padding = _preprocess_padding(padding)
    x, tf_data_format = _preprocess_conv1d_input(x, data_format)
    x = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=(dilation_rate,),
        strides=(strides,),
        padding=padding,
        data_format=tf_data_format)
    if data_format == 'channels_first' and tf_data_format == 'NWC':
        x = tf.transpose(x, (0, 2, 1))  # NWC -> NCW
    return x


def conv2d(x, kernel, strides=(1, 1), padding='valid',
           data_format=None, dilation_rate=(1, 1)):
    """2D convolution.

    # Arguments
        x: Tensor or variable.
        kernel: kernel tensor.
        strides: strides tuple.
        padding: string, `"same"`, `"valid"` or `"causal"`.
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
    data_format = normalize_data_format(data_format)
    kernel_shape = kernel.get_shape().as_list()

    if padding == 'causal':
        if data_format != 'channels_last':
            raise ValueError('When using causal padding in `conv1d`, '
                             '`data_format` must be "channels_last" '
                             '(temporal data).')
        left_pad = dilation_rate[0] * (kernel_shape[0] - 1)
        lower_pad = dilation_rate[1] * (kernel_shape[1] - 1)
        x = spatial_2d_padding(x, ((left_pad, 0), (0, lower_pad)))
        padding = 'valid'
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
