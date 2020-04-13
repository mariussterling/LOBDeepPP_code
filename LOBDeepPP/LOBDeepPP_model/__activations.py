from keras import layers, constraints
from keras import backend as K


class PReLU2(layers.PReLU):
    def __init__(self, alpha_initializer='zeros', alpha_regularizer=None,
                 alpha_constraint=None, shared_axes=None, **kwargs):
        super(layers.PReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = layers.initializers.get(alpha_initializer)
        self.alpha_regularizer = layers.regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.__name__ = 'PReLU'
        if shared_axes is None:
            self.shared_axes = None
        else:
            self.shared_axes = layers.to_list(shared_axes, allow_tuple=True)


class LeakyReLU2(layers.LeakyReLU):
    def __init__(self, alpha=0.3, **kwargs):
        super(layers.LeakyReLU, self).__init__(**kwargs)
        self.__name__ = 'LeakyReLU2'
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
