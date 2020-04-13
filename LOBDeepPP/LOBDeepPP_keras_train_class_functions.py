from tqdm import tqdm
from keras import backend
import tensorflow as tf
import numpy as np
from keras import layers, models


def get_layer_names(model):
    res = []
    for i in model.layers:
        if isinstance(i, models.Model):
            for j in get_layer_names(i):
                res.append((i.name, j))
        else:
            res.append(i.name)
    return res


def contains_subnetwork(model):
    if [True for l in model.layers if isinstance(l, models.Model)]:
        return True
    else:
        return False


def set_layer_weights(layer0, layer1):
    if isinstance(layer0, layers.InputLayer):
        return
    elif isinstance(layer1, layers.TimeDistributed) and \
            isinstance(layer1.layer, layers.LocallyConnected1D) and \
            not isinstance(layer0, layers.TimeDistributed):
        s = layer1.input.get_shape().as_list()
        weight, bias = layer1.get_weights()
        weight2 = np.concatenate([weight for _ in range(s[1])], axis=0)
        bias2 = np.concatenate([np.expand_dims(bias, axis=0) for
                                _ in range(s[1])], axis=0)
        layer0.set_weights([weight2, bias2])
    else:
        layer0.set_weights(layer1.get_weights())


def set_model_weights(model0, model1):
    backend.get_session().run(tf.global_variables_initializer())
    if contains_subnetwork(model1) and \
            contains_subnetwork(model0):
        for l in model0.layers:
            set_layer_weights(
                model0.get_layer(l.name),
                model1.get_layer(l.name))
    elif (not contains_subnetwork(model1)) and \
            contains_subnetwork(model0):
        raise NotImplementedError()
    elif contains_subnetwork(model1) and \
            (not contains_subnetwork(model0)):
        layer_names = get_layer_names(model1)
        print(f'\nSetting weights of model {model0.name} to be the same as '
              f'model {model1.name}')
        for l in tqdm(model0.layers):
            fit = [i for i in layer_names if
                   isinstance(i, str) and l.name == i
                   or isinstance(i, tuple) and l.name == i[1]]
            assert len(fit) == 1, f'len(fit) == 1, but it is {len(fit)}'
            fit = fit[0]
            if isinstance(fit, tuple):
                set_layer_weights(
                    model0.get_layer(l.name),
                    model1.get_layer(fit[0]).get_layer(fit[1]))
            elif isinstance(fit, str):
                set_layer_weights(
                    model0.get_layer(l.name),
                    model1.get_layer(l.name))
            else:
                raise NotImplementedError
    elif (not contains_subnetwork(model1)) and \
            (not contains_subnetwork(model0)):
        for l in tqdm(model0.layers):
            set_layer_weights(
                model0.get_layer(l.name),
                model1.get_layer(l.name))
    else:
        raise ValueError
    return model0
