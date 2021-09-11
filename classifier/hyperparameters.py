import numpy as np

class Hyperparameters():
    def __init__(self, lr_power, decay, reduce_lr_factor, reduce_lr_patience, filters_multipliers=None, filters=None,
                 kernel_size_multipliers=None, kernel_sizes=None, extra_dense_layers_no=None,
                 dense_outputs=None, depth=None, lstm_units=None):
        self.filters_multipliers = filters_multipliers
        self.filters = filters
        self.kernel_size_multipliers = kernel_size_multipliers
        self.kernel_sizes = kernel_sizes
        self.extra_dense_layers_no = extra_dense_layers_no
        self.dense_outputs = dense_outputs
        self.depth = depth
        self.lstm_units = lstm_units

        self.lr = np.float_power(10, lr_power)
        self.decay = decay
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
