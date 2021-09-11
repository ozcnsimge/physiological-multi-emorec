import kapre
import keras
import numpy as np

from classifier.classifier import Classifier, focal_loss


class ClassifierStresnet(Classifier):

    def __init__(self, output_directory, input_shapes, sampling_rates, ndft_arr, nb_classes, verbose=True,
                 hyperparameters=None, model_init=None):
        self.sampling_rates = sampling_rates
        self.ndft_arr = ndft_arr
        print(verbose)

        super().__init__(output_directory, input_shapes, nb_classes, verbose=verbose,
                         hyperparameters=hyperparameters, model_init=model_init)

    def one_chennel_resnet(self, input_shape, num_filters, num_res_blocks, cnn_per_res,
                           kernel_sizes, max_filters, pool_size, pool_stride_size):
        my_input = keras.layers.Input(shape=(input_shape))
        for i in np.arange(num_res_blocks):
            if (i == 0):
                block_input = my_input
                x = keras.layers.BatchNormalization()(block_input)
            else:
                block_input = x
            for j in np.arange(cnn_per_res):
                x = keras.layers.Conv1D(num_filters, kernel_sizes[j], padding='same')(x)
                x = keras.layers.BatchNormalization()(x)
                if (j < cnn_per_res - 1):
                    x = keras.layers.Activation('relu')(x)
            is_expand_channels = not (input_shape[0] == num_filters)
            if is_expand_channels:
                res_conn = keras.layers.Conv1D(num_filters, 1, padding='same')(block_input)
                res_conn = keras.layers.BatchNormalization()(res_conn)
            else:
                res_conn = keras.layers.BatchNormalization()(block_input)
            x = keras.layers.add([res_conn, x])
            x = keras.layers.Activation('relu')(x)
            if (i < 4):  # perform pooling only for the first 4 layers
                x = keras.layers.AveragePooling1D(pool_size=pool_size, strides=pool_stride_size)(x)
            num_filters = 2 * num_filters
            if max_filters < num_filters:
                num_filters = max_filters

        x = keras.layers.BatchNormalization()(x)
        return my_input, x

    def spectro_layer_mid(self, input_x, sampling_rate, ndft):
        l2_lambda = .001
        fmin = 0.0
        n_hop = ndft // 2
        fmax = sampling_rate // 2
        num_filters = 32
        x = keras.layers.Permute((2, 1))(input_x)
        x = kapre.time_frequency.Spectrogram(n_dft=ndft, n_hop=n_hop,
                                             image_data_format='channels_last', return_decibel_spectrogram=True)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        return x

    def build_model(self, input_shapes, nb_classes, hyperparameters):
        drop_rate = .25
        inputs = []
        l2_lambda = .001
        channel_outputs = []
        num_filters = 16
        num_channels = len(input_shapes)

        num_res_blocks = hyperparameters.depth if hyperparameters and hyperparameters.depth else 3
        cnn_per_res = 3
        kernel_sizes = [12, 12, 9, 9, 5, 5]
        max_filters = 64
        pool_size = 1
        pool_stride_size = 1

        if num_res_blocks > 3:
            num_res_blocks = 3

        # chgannel specific residual layers (time-domain)
        for i in np.arange(num_channels):
            max_filters_for_channel = int(hyperparameters.kernel_size_multipliers[
                                              i] * max_filters) if hyperparameters and hyperparameters.kernel_size_multipliers else max_filters
            num_filters_for_channel = int(hyperparameters.kernel_size_multipliers[
                                              i] * num_filters) if hyperparameters and hyperparameters.kernel_size_multipliers else num_filters

            kernel_sizes_multiplier = hyperparameters.kernel_size_multipliers[
                i] if hyperparameters and hyperparameters.kernel_size_multipliers else 1

            kernel_sizes_for_channel = [int(x * kernel_sizes_multiplier) for x in kernel_sizes]

            channel_resnet_input, channel_resnet_out = self.one_chennel_resnet(
                input_shapes[i], num_filters_for_channel, num_res_blocks, cnn_per_res,
                kernel_sizes_for_channel, max_filters_for_channel, pool_size, pool_stride_size)
            channel_resnet_out = keras.layers.Flatten()(channel_resnet_out)
            channel_outputs.append(channel_resnet_out)
            inputs.append(channel_resnet_input)

        # chgannel spectral layers (frequency-domain)
        spectral_outputs = []
        num_filters = 16
        COUNTER = 0
        for x in inputs:
            channel_out = self.spectro_layer_mid(x, self.sampling_rates[COUNTER], ndft=self.ndft_arr[COUNTER])
            channel_out = keras.layers.Flatten()(channel_out)
            spectral_outputs.append(channel_out)
            COUNTER = COUNTER + 1

        # concateante the chgannel specific residual layers
        x = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]

        # join time-domain and frequnecy domain fully-conencted layers
        s = keras.layers.concatenate(spectral_outputs, axis=-1) if len(spectral_outputs) > 1 else spectral_outputs[0]

        x = keras.layers.concatenate([s, x])
        x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = keras.layers.Dropout(drop_rate)(x)
        output = keras.layers.Dense(nb_classes, activation="softmax")(x)

        model = keras.models.Model(inputs=inputs, outputs=output)
        optimizer = self.get_optimizer()
        # uncomment to use focal loss
        # model.compile(optimizer=optimizer, loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        return model
