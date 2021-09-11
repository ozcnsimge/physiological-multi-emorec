import keras
from keras.layers import Input, Conv1D, Flatten, Dense, LSTM, TimeDistributed, MaxPooling1D

from classifier.classifier import Classifier, reshape_samples


class ClassifierCnnLstm(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []
        output_channel = []
        drop_rate = .25
        l2_lambda = .001

        if hyperparameters:
            lstm_units = hyperparameters.lstm_units
            filters_multipliers = hyperparameters.filters_multipliers
            kernel_size_multipliers = hyperparameters.kernel_size_multipliers
        else:
            lstm_units = [64] * len(input_shapes)
            filters_multipliers = [1] * len(input_shapes)
            kernel_size_multipliers = [0.5] * len(input_shapes)

        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = Input(shape=(None, round(input_shape[0] / 2), 1), name=f"input_{channel_id}")
            input_layers.append(input_layer)

            filters_1 = int(filters_multipliers[channel_id] * 32)
            filters_2 = int(filters_multipliers[channel_id] * 64)
            kernel_size_1 = int(kernel_size_multipliers[channel_id] * 2)
            kernel_size_2 = int(kernel_size_multipliers[channel_id] * 4)

            # length of convolution window (kernel size) cannot be larger than number of steps
            conv_layer = TimeDistributed(Conv1D(filters=filters_1, kernel_size=kernel_size_1))(input_layer)
            conv_layer = TimeDistributed(MaxPooling1D(pool_size=2))(conv_layer)

            conv_layer = TimeDistributed(Conv1D(filters=filters_2, kernel_size=kernel_size_2))(conv_layer)
            conv_layer = TimeDistributed(MaxPooling1D(pool_size=2))(conv_layer)

            flatten_layer = TimeDistributed(Flatten())(conv_layer)

            dense_layer = TimeDistributed(Dense(lstm_units[channel_id]))(flatten_layer)

            lstm_layer = LSTM(lstm_units[channel_id])(dense_layer)

            output_channel.append(lstm_layer)

        if len(output_channel) == 1:
            flatten_layer = output_channel[0]
        else:
            flatten_layer = keras.layers.concatenate(output_channel, axis=-1)

        x = keras.layers.Dropout(drop_rate)(flatten_layer)
        x = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_lambda))(x)
        x = keras.layers.Dropout(drop_rate)(x)
        output_layer = Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size=16, nb_epochs=2, x_test=None, shuffle=True):
        x_train = reshape_samples(x_train)
        x_val = reshape_samples(x_val)
        if x_test is not None:
            x_test = reshape_samples(x_test)

        return super().fit(x_train, y_train, x_val, y_val, y_true, batch_size=batch_size, nb_epochs=nb_epochs,
                           x_test=x_test)