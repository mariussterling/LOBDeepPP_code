from keras import models
from .__metric import mde_perc
from .__LOB_models_features import fe0a, fe1o
from .__LOB_models_output2D import output_model_askbid
from .__LOB_models_tcn import tcn2d_model


# %% Model 17a
def LOB_keras_model17a(*, inp=None, output_shape=None, params=None,
                       **kwargs):
    """Model 17a
    """
    interpretable = params.get('interpretable', None)
    if interpretable is None:
        interpretable = kwargs.get('interpretable', False)

    lag, levels, channels = inp.get_shape().as_list()[1:]
    out = inp

    out = fe0a(out, params, interpretable, 2)
    out = tcn2d_model(out, params['tcn']['base'], 'tcn_base', interpretable)
    out = output_model_askbid(
        out, params, output_shape, interpretable, interpretable_nested=False)
    model = models.Model(inputs=inp, outputs=out, name='LOB_keras_model17a')
    model.compile(loss='mse', optimizer='adam',
                  metrics=['logcosh', 'mean_absolute_percentage_error',
                           mde_perc])
    return model


# %% Model 17b
def LOB_keras_model17b(*, inp=None, output_shape=None, params=None,
                       **kwargs):
    """Model 17b
    """
    interpretable = params.get('interpretable', None)
    if interpretable is None:
        interpretable = kwargs.get('interpretable', False)

    lag, levels, channels = inp.get_shape().as_list()[1:]
    out = inp

    out = fe1o(out, params, interpretable)
    out = tcn2d_model(out, params['tcn']['base'], 'tcn_base', interpretable)
    out = output_model_askbid(
        out, params, output_shape, interpretable, interpretable_nested=False)
    model = models.Model(inputs=inp, outputs=out, name='LOB_keras_model17b')
    model.compile(loss='mse', optimizer='adam',
                  metrics=['logcosh', 'mean_absolute_percentage_error',
                           mde_perc])
    return model
