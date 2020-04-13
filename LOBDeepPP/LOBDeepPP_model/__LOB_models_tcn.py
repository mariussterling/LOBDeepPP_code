
from keras import layers, models
from .LOBDeepPP_TCN import TCN2D


def tcn2d_model(inp, tcn_params, tcn_name, interpretable):
    if interpretable:
        out = inp
    else:
        out_inp = layers.InputLayer(
            input_shape=inp.get_shape().as_list()[1:], name='out_inp')
        out = out_inp.output
    out = TCN2D(
        nb_filters=tcn_params['filters'],
        dilations=tcn_params['dilations'],
        kernel_size=tcn_params['kernel_size'],
        dropout_rate=tcn_params['dropout_rate'],
        bias_constraint=None,
        padding=tcn_params['padding'],
        nb_stacks=tcn_params['nb_stacks'],
        lambda_layer=False,
        return_sequences=True,
        use_skip_connections=tcn_params['use_skip_connections'],
        name=tcn_name)(out)
    if interpretable:
        return out
    else:
        return models.Model(
            inputs=out_inp.input,
            outputs=out,
            name=tcn_name)(inp)
