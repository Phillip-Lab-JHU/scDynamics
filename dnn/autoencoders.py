# Chanhong Min <cmin11@jhmi.edu>

# Copyright 2023 The Phillip tiME Lab at the Johns Hopkins University
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/Phillip-Lab-JHU/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Architectures of DNN autoencoder"""

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

def Res_Conv1D_LSTM(duration, coor_dim=3, dimension=128):
    '''
    Accepts X, Y time series as input, performs 1D convs with an LSTM layer
    and upsamples as an autoencoder.
    Parameters
    ----------
    t : integer.
        length of time series.
    n_channels : integer.
        number of channels in data.
    Returns
    -------
    model : keras model object.
    '''
    tf.keras.backend.clear_session()
    def res_conv_block(input_layer, kernel_size, num_filters, dropout=0.1, batch_norm=True):
        conv = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', strides=1)(input_layer)
        # padding same: input shape = output shape with zero padding (only when stride = 1)
        # if stride >=2, output shape = input shape / stride
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_series, duration, dimension)에서 dimension 방향으로 normalize
        conv = layers.Activation('relu')(conv)

        conv = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv1D(filters=num_filters, kernel_size=1, padding='same', strides=1)(input_layer)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    x = layers.Input(shape=(duration, coor_dim))
    cv1 = res_conv_block(x, kernel_size=3, num_filters=16, dropout=0.1, batch_norm=True)
    p1 = layers.MaxPooling1D(pool_size=2)(cv1)
    cv2 = res_conv_block(p1, kernel_size=3, num_filters=32, dropout=0.1, batch_norm=True)
    p2 = layers.MaxPooling1D(pool_size=2)(cv2)
    cv3 = res_conv_block(p2, kernel_size=3, num_filters=64, dropout=0.1, batch_norm=True)

    lstm = layers.LSTM(256, dropout=0.2, recurrent_dropout=0)(cv3)

    d1 = layers.Dense(dimension, activation='relu')(lstm)
    do1 = layers.Dropout(rate=0.2)(d1)
    d2 = layers.Dense(cv3.shape[1] * cv3.shape[2], activation='relu')(do1)
    do2 = layers.Dropout(rate=0.2)(d2)

    rs = layers.Reshape(target_shape=(cv3.shape[1], cv3.shape[2]))(do2)

    uc1 = res_conv_block(rs, kernel_size=3, num_filters=64, dropout=0.1, batch_norm=True)
    us1 = layers.UpSampling1D(size=2)(uc1)
    uc2 = res_conv_block(us1, kernel_size=3, num_filters=32, dropout=0.1, batch_norm=True)
    us2 = layers.UpSampling1D(size=2)(uc2)
    uc3 = res_conv_block(us2, kernel_size=3, num_filters=16, dropout=0.1, batch_norm=True)
    y = layers.Conv1D(filters=coor_dim, kernel_size=3, padding='same', activation=None)(uc3)

    model = models.Model(x, y)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1), metrics = ['accuracy'])
    model.summary()

    return model


def Temporal_Conv1D_2D(duration, coor_dim=3, dimension=128):
    tf.keras.backend.clear_session()
    def causal_res_conv1d_block(input_layer, kernel_size, num_filters, dilation_rate):
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            input_layer)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)

        # if input_layer.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv1D(filters=num_filters, kernel_size=1, padding='same')(input_layer)
        shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, x])

        return residual

    def res_conv2d_block(image, kernel_size, num_filters, dropout=0.1, batch_norm=True):
        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(image)
        # padding same: input shape = output shape with zero padding (only when stride = 1)
        # stride >=2 이면 output shape = input shape / stride
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('relu')(conv)

        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same', strides=1)(image)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    assert coor_dim >= 2, "coordinate dimension must be larger than 1"
    if coor_dim < 3:
        kernel_size_2d = 2
    elif coor_dim >= 3:
        kernel_size_2d = 3

    x = layers.Input(shape=(duration, coor_dim))

    x_2d = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)
    cv1_2d = res_conv2d_block(x_2d, kernel_size=kernel_size_2d, num_filters=16, dropout=0.1, batch_norm=True)
    cv2_2d = res_conv2d_block(cv1_2d, kernel_size=kernel_size_2d, num_filters=32, dropout=0.1, batch_norm=True)
    f_2d = layers.Flatten()(cv2_2d)
    drop_2d = layers.Dropout(0.1)(f_2d)
    d1_2d = layers.Dense(100, activation='relu')(drop_2d)

    cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)
    p1 = layers.MaxPooling1D(pool_size=2)(cv1)
    cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)
    p2 = layers.MaxPooling1D(pool_size=2)(cv2)
    cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)
    f = layers.Flatten()(cv3)
    drop = layers.Dropout(0.1)(f)
    d1 = layers.Dense(100, activation='relu')(drop)

    concat = layers.Concatenate(name='Concatenate')([d1_2d, d1])
    concat_d1 = layers.Dense(dimension, activation='relu')(concat)
    concat_do1 = layers.Dropout(rate=0.1)(concat_d1)
    concat_d2 = layers.Dense(cv3.shape[1] * cv3.shape[2], activation='relu')(concat_do1)
    concat_do2 = layers.Dropout(rate=0.1)(concat_d2)
    concat_rs = layers.Reshape(target_shape=(cv3.shape[1], cv3.shape[2]))(concat_do2)

    uc1 = causal_res_conv1d_block(concat_rs, kernel_size=3, num_filters=64, dilation_rate=4)
    us1 = layers.UpSampling1D(size=2)(uc1)
    uc2 = causal_res_conv1d_block(us1, kernel_size=3, num_filters=32, dilation_rate=2)
    us2 = layers.UpSampling1D(size=2)(uc2)
    uc3 = causal_res_conv1d_block(us2, kernel_size=3, num_filters=16, dilation_rate=1)
    y = layers.Conv1D(filters=coor_dim, kernel_size=3, padding='same', activation=None)(uc3)


    model = models.Model(x, y)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1), metrics=['accuracy'])
    model.summary()

    return model


def set_duration_for_autoencoder(trajectories:dict[int, np.array], duration:int, dim:int) -> dict[int, np.array]:
    if duration % 4 == 0:
        new_duration = duration
        trajectories_new = trajectories

    if duration % 4 == 1:
        new_duration = duration - 1
        trajectories_new = {}
        for traj_idx, traj in trajectories.items():
            new_traj = np.zeros(shape=(new_duration, dim))
            new_traj = traj[:-1, :]  # Remove the last coordinate
            trajectories_new[traj_idx] = new_traj

    if duration % 4 == 2:
        new_duration = duration + 2
        trajectories_new = {}
        for traj_idx, traj in trajectories.items():
            new_traj = np.zeros(shape=(new_duration, dim))
            new_traj[:-2, :] = traj
            new_traj[-2:, :] = traj[-1]  # Duplicate last coordinate twice
            trajectories_new[traj_idx] = new_traj

    if duration % 4 == 3:
        new_duration = duration + 1
        trajectories_new = {}
        for traj_idx, traj in trajectories.items():
            new_traj = np.zeros(shape=(new_duration, dim))
            new_traj[:-1, :] = traj
            new_traj[-1, :] = traj[-1]  # Duplicate last coordinate once
            trajectories_new[traj_idx] = new_traj

    return trajectories_new, new_duration