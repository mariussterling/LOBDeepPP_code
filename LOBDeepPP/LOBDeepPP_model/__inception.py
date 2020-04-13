from keras import layers


def inception(inp, filters, name, bias_constraint=None):
    out1 = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',  # 'relu',
        name=name + '_conv3_1')(inp)
    out1 = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',  # 'relu',
        name=name + '_conv3')(out1)
    out2 = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',  # 'relu',
        name=name + '_conv5_1')(inp)
    out2 = layers.Conv1D(
        filters=filters,
        kernel_size=5,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',  # 'relu',
        name=name + '_conv5')(out2)
    out3 = layers.MaxPool1D(
        pool_size=3,
        strides=1,
        padding='same',
        name=name + '_mxpool3')(inp)
    out3 = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',  # 'relu',
        name=name + '_mxpool3_1')(out3)
    out4 = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',  # 'relu',
        name=name + '_skip')(inp)
    out = layers.Concatenate(axis=2, name=name + '_concatenate')([
        out1, out2, out3, out4])
    return out


def inception2D(inp, filters, name, bias_constraint=None):
    out1 = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',
        name=name + '_conv3_1')(inp)
    out1 = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',
        name=name + '_conv3')(out1)
    out2 = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',
        name=name + '_conv5_1')(inp)
    out2 = layers.Conv2D(
        filters=filters,
        kernel_size=5,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',
        name=name + '_conv5')(out2)
    out3 = layers.MaxPool2D(
        pool_size=3,
        strides=1,
        padding='same',
        name=name + '_mxpool3')(inp)
    out3 = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',
        name=name + '_mxpool3_1')(out3)
    out4 = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding='same',
        bias_constraint=bias_constraint,
        activation='relu',
        name=name + '_skip')(inp)
    out = layers.Concatenate(axis=3, name=name + '_concatenate')([
        out1, out2, out3, out4])
    return out
