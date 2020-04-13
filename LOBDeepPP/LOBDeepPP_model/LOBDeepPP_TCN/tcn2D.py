# Code for TCN was taken from https://github.com/philipperemy/keras-tcn

import keras.backend as K
import keras.layers
from keras.layers import Activation, Lambda
from keras.layers import Conv2D
from keras.layers import SpatialDropout2D
from keras.layers import BatchNormalization


from .keras_separable_conv1d_replace import separable_conv2d, conv2d
K.separable_conv2d = separable_conv2d
K.conv2d = conv2d


def residual_block(x, dilation_rate, nb_filters, kernel_size, padding,
                   dropout_rate=0, name='tcn', **kwargs):
    # # type: (Layer, int, int, int, str, float) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual
            block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or
            'causal'.
        dropout_rate: Float between 0 and 1. Fraction of the input units
            to drop.

    Returns:
        A tuple where the first element is the residual model layer, and the
        second is the skip connection.
    """
    prev_x = x
    for k in range(2):
        x = Conv2D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=padding,
            bias_constraint=kwargs.get('bias_constraint', None),
            name=f'{name}_dil{dilation_rate[0]}_resBlock_{k}_conv2D')(x)
        x = BatchNormalization(
            name=f'{name}_dil{dilation_rate[0]}_resBlock_{k}_bn')(x)
        x = Activation(
            'relu',
            name=f'{name}_dil{dilation_rate[0]}_resBlock_{k}_relu')(x)
        x = SpatialDropout2D(
            rate=dropout_rate,
            name=f'{name}_dil{dilation_rate[0]}_resBlock_{k}_dropout')(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='valid',
        activation='relu',
        bias_constraint=kwargs.get('bias_constraint', None),
        kernel_constraint=kwargs.get('kernel_constraint', None),
        name=f'{name}_dil{dilation_rate[0]}_conv1D')(prev_x)
    res_x = keras.layers.add([prev_x, x],
                             name=f'{name}_dil{dilation_rate[0]}_add')
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class TCN2D:
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional
                layers.
            kernel_size: The size of the kernel to use in each convolutional
                layer.
            dilations: The list of the dilations.
                Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers,
                'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections
                from input to each residual block.
            return_sequences: Boolean. Whether to return the last output
                in the output sequence, or the full sequence.
            dropout_rate: Float between 0 and 1. Fraction of the input units
                to drop.
            name: Name of the model. Useful when having multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=[2, 2],
                 nb_stacks=1,
                 dilations=[(1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1)],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 name='tcn2d',
                 conv_layer=None,
                 **kwargs):
        self.name = name
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        if isinstance(nb_filters, int):
            self.nb_filters = [nb_filters for _ in dilations]
        elif len(nb_filters) == len(dilations):
            self.nb_filters = nb_filters
            # raise NotImplementedError('nb_filters input as list')
        else:
            raise ValueError('nb_filters')
        self.padding = padding
        self.lambda_layer = kwargs.get('lambda_layer', True)
        self.kwargs = kwargs

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible \
                             for this layer.")

        if self.nb_stacks != 1 and not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN2D(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN2D(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 \
                  (pip install keras-tcn==2.1.2).')
            raise Exception()

    def __call__(self, inputs):
        x = inputs
        # 2D FCN.
        x = Conv2D(
            filters=self.nb_filters[0],
            kernel_size=1,
            padding=self.padding,
            activation='relu',
            bias_constraint=self.kwargs.get('bias_constraint', None),
            name=f'{self.name}_input_conv2d')(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                x, skip_out = residual_block(
                    x,
                    dilation_rate=d,
                    nb_filters=self.nb_filters[i],
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    dropout_rate=self.dropout_rate,
                    name=self.name,
                    **self.kwargs)
                skip_connections.append(skip_out)
        if self.use_skip_connections:
            if len(skip_connections) > 1:
                x = keras.layers.add(
                    skip_connections,
                    name=f'{self.name}_dil{self.dilations[-1][0]}_add')
            else:
                x = skip_connections[0]
        if not self.return_sequences:
            if self.lambda_layer:
                x = Lambda(lambda tt: tt[:, -1, :],
                           name=f'{self.name}_output_lambda')(x)
            else:
                x = keras.layers.Cropping2D(
                    cropping=((x.shape[1].value - 1, 0), (0, 0)),
                    name=f'{self.name}_output_reshape_cropping')(x)
                x = keras.layers.Reshape(
                    target_shape=[i.value for i in x.shape[2:]],
                    name=f'{self.name}_output_reshape_reshape')(x)
        return x
