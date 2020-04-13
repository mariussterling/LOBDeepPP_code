from keras import layers, models
from .__inception import inception2D

bn_axis = -1


def fe0a(input, params, interpretable, L):
    if interpretable:
        out = input
    else:
        inp = layers.InputLayer(input_shape=input.get_shape().as_list()[1:],
                                name='inp0')
        out = inp.output
    # out = layers.BatchNormalization(axis=bn_axis, name='fe_bn0')(out)
    # out = LayerNormalization(name='fe_ln0')(out)
    # Feature generation: ind. order-wise aggregation
    if not interpretable:
        out1 = layers.TimeDistributed(layers.LocallyConnected1D(
            filters=params['features']['filters'][0],
            kernel_size=2,
            strides=2,
            padding='valid',
            activation='relu',
            bias_constraint=eval(params['features']['bias_constraint']),
        ), name='fe_order_l0')(out)
    else:
        out1 = layers.LocallyConnected2D(
            filters=params['features']['filters'][0],
            kernel_size=(1, 2),
            strides=(1, 2),
            padding='valid',
            activation='relu',
            bias_constraint=eval(params['features']['bias_constraint']),
            name='fe_order_l0')(out)
    out1 = layers.BatchNormalization(axis=bn_axis, name='fe_bn1')(out1)
    # Feature generation: aggregation of neighbouring bid/ask orders
    out1a = layers.Conv2D(
        filters=params['features']['filters'][1],
        kernel_size=(1, 2),
        dilation_rate=(1, 2),
        padding='valid',
        activation='relu',
        bias_constraint=eval(params['features']['bias_constraint']),
        name='fe_order_l1a')(out1)

    # Feature generation: aggregation of bid/ask side
    out1b = layers.Conv2D(
        filters=params['features']['filters'][1],
        kernel_size=(1, out1.get_shape().as_list()[2] // 2),
        dilation_rate=(1, 2),
        padding='valid',
        activation='relu',
        bias_constraint=eval(params['features']['bias_constraint']),
        name='fe_order_l1b')(out1)
    out1c = layers.Conv2D(
        filters=params['features']['filters'][1],
        kernel_size=1,
        padding='valid',
        activation='relu',
        bias_constraint=eval(params['features']['bias_constraint']),
        name='fe_order_l1c')(out1)
    out = layers.Concatenate(axis=2, name='fe_concatenate_l2')([
        out1a, out1b, out1c])
    out = layers.BatchNormalization(axis=bn_axis, name='fe_out0_bn')(out)
    dim = out.get_shape().as_list()
    out = layers.Reshape([dim[1], 4, dim[2] * dim[3] // 4],
                         name=f'fe_out0_reshape')(out)
    out = inception2D(
        out, 128 // 4, f'fe_out0_inc',
        eval(params['output']['bias_constraint']))
    if interpretable:
        return out
    else:
        return models.Model(inputs=inp.input, outputs=out, name='fe0a')(input)


def fe1o(input, params, interpretable):
    if interpretable:
        out = input
    else:
        inp = layers.InputLayer(input_shape=input.get_shape().as_list()[1:],
                                name='inp0')
        out = inp.output
    dim = out.get_shape().as_list()
    out = layers.Reshape([dim[1], dim[2] // 2, 2])(out)
    out = layers.Lambda(lambda x: x[:, :, :2, :1],
                        name=f'fe_lambda_ask_bid_price')(out)
    out = layers.BatchNormalization(axis=bn_axis, name='fe_bn0')(out)

    out1 = layers.Conv2D(
        filters=params['features']['filters'][1],
        kernel_size=(1, 1),
        padding='valid',
        activation='relu',  # 'relu',
        bias_constraint=eval(params['features']['bias_constraint']),
        name='fe_order_l1a')(out)
    out2 = layers.Conv2D(
        filters=params['features']['filters'][1],
        kernel_size=(1, 2),
        padding='same',
        activation='relu',  # 'relu',
        bias_constraint=eval(params['features']['bias_constraint']),
        name='fe_order_l1b')(out)
    out = layers.Concatenate(axis=-2, name='fe_order_concat')([out1, out2])
    out = layers.BatchNormalization(axis=bn_axis, name='fe_out_bn')(out)
    out = inception2D(out, 128 // 4, f'fe_out_inc',
                      eval(params['output']['bias_constraint']))
    if interpretable:
        return out
    else:
        return models.Model(inputs=inp.input, outputs=out, name='fe0o')(input)
