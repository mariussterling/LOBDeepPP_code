from keras import backend as K


def mde(y_true, y_pred):
    """
    Mean direction error.

    Parameters
    ----------
    y_true : tensor
        Tensor of true values.
    y_pred : tensor
        Tensor of predicted values.

    Returns
    -------
    float
        Returns MDE.

    """
    return 1 - K.mean(K.equal(K.sign(y_true), K.sign(y_pred)))


def mde_perc(y_true, y_pred):
    """
    Mean direction error.

    Parameters
    ----------
    y_true : tensor
        Tensor of true values.
    y_pred : tensor
        Tensor of predicted values.

    Returns
    -------
    float
        Returns MDE.

    """
    return 100 * mde(y_true, y_pred)


def r2(y_true, y_pred):
    """R squared.

    Parameters
    ----------
    y_true : tensor
        Tensor of true values.
    y_pred : tensor
        Tensor of predicted values.

    Returns
    -------
    float
        Returns R-squared
    """
    ss_res = K.mean(K.pow(y_pred - y_true, 2))
    ss_tot = K.mean(K.pow(y_true - K.mean(y_true), 2))
    return 1 - ss_res / ss_tot
