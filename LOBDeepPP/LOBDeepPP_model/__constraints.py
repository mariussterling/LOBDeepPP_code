from keras import constraints
from keras import backend as K


class NonPos(constraints.Constraint):
    """Constrains the weights to be non-positive.
    """

    def __call__(self, w):
        w *= K.cast(K.less_equal(w, 0.), K.floatx())
        return w
