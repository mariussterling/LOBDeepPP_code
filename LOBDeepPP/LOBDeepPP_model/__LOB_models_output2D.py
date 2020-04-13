from keras import layers, models

from .__activations import PReLU2


def output_model_askbid(inp, params, output_shape, interpretable, **kwargs):
    if interpretable:
        out = inp
    else:
        out_inp = layers.InputLayer(
            input_shape=inp.get_shape().as_list()[1:],
            name='out_inp')
        out = out_inp.output
    out = layers.Cropping2D(
        cropping=((out.shape[1].value - 1, 0), (0, 0)),
        name=f'out_cropping')(out)
    out = layers.Reshape(
        target_shape=[i.value for i in out.shape[2:]],
        name='out_reshape')(out)
    out_ask = output_model_b(
        out, params, output_shape[0],
        interpretable=kwargs.get('interpretable_nested', True),
        name='ask')
    out_bid = output_model_b(
        out, params, output_shape[0],
        interpretable=kwargs.get('interpretable_nested', True),
        name='bid')
    out = layers.concatenate([out_ask, out_bid], name='out_concatenate')
    if interpretable:
        return out
    else:
        return models.Model(inputs=out_inp.input, outputs=out, name='out')(inp)


def output_model_b(inp, params, output_shape, interpretable, name=''):
    # h = params.get('output').get('h', output_shape)
    if interpretable:
        out = inp
    else:
        out_inp = layers.InputLayer(
            input_shape=inp.get_shape().as_list()[1:],
            name=f'out_{name}_inp')
        out = out_inp.output
    filters = params['output'].get('filters', None)

    for i, f in enumerate(filters):
        out = layers.Dense(f, name=f'out_{name}_dense{i}')(out)
        out = PReLU2(name=f'out_{name}_dense{i}_relu')(out)
        out = layers.BatchNormalization(name=f'out_{name}_dense{i}_bn')(out)
    out = layers.Flatten(name=f'out_{name}_flatten')(out)

    out_p = layers.Dense(
        output_shape,
        name=f'out_{name}_out_pos')(out)
    out_p = PReLU2(name=f'out_{name}_out_pos_relu')(out_p)
    out_n = layers.Lambda(
        lambda x: x * -1,
        name=f'out_{name}_out_neg0')(out)
    out_n = layers.Dense(
        output_shape,
        # activation='relu',
        name=f'out_{name}_out_neg')(out_n)
    out_n = PReLU2(name=f'out_{name}_out_neg_relu')(out_n)

    out = layers.Subtract(name=f'out_{name}_out')([out_p, out_n])
    out = layers.Reshape(
        target_shape=out.get_shape().as_list()[1:] + [1],
        name=f'out_{name}_reshape')(out)
    if interpretable:
        return out
    else:
        return models.Model(
            inputs=out_inp.input,
            outputs=out,
            name=f'out_{name}'
        )(inp)
