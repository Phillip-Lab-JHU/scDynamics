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
"""Architectures of DNN classification"""

from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

def Temporal_Conv1D_2D_classifier(duration, coor_dim, n_classes):
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
    concat_d1 = layers.Dense(128, activation='relu')(concat)
    concat_do1 = layers.Dropout(rate=0.1)(concat_d1)
    concat_d2 = layers.Dense(64, activation='relu')(concat_do1)
    concat_do2 = layers.Dropout(rate=0.1)(concat_d2)
    concat_d3 = layers.Dense(32, activation='relu')(concat_do2)
    concat_do3 = layers.Dropout(rate=0.1)(concat_d3)
    concat_d4 = layers.Dense(16, activation='relu')(concat_do3)
    concat_do4 = layers.Dropout(rate=0.1)(concat_d4)

    if n_classes == 2:  # Binary classification
        y = layers.Dense(1, activation='sigmoid')(concat_do4)
        model = models.Model(x, y)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
    elif n_classes >= 3:  # Multi-class classification
        y = layers.Dense(n_classes, activation='softmax')(concat_do4)
        model = models.Model(x, y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    elif n_classes == 0:  # Regression
        y = layers.Dense(1, activation='linear')(concat_do4)
        model = models.Model(x, y)
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      # metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
                      )

    #model.summary()

    return model

def GC_classifier(duration, coor_dim, n_classes):
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


    x = layers.Input(shape=(duration, coor_dim))

    cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)  # (None, duration, 16)
    p1 = layers.MaxPooling1D(pool_size=2)(cv1)  # (None, duration/2, 16)
    cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)  # (None, duration/2, 32)
    p2 = layers.MaxPooling1D(pool_size=2)(cv2)  # (None, duration/4, 32)
    cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)  # (None, duration/4, 64)
    x_prime = layers.Conv1D(filters=coor_dim, kernel_size=1, padding='same')(cv3)  # (None, duration/4, coor_dim)




    cv1_f = layers.Dense(100, activation=None)(layers.Flatten()(cv1))
    #cv1_f = layers.LayerNormalization(epsilon=1e-6, axis=-1)(cv1_f)
    cv2_f = layers.Dense(100, activation=None)(layers.Flatten()(cv2))
    #cv2_f = layers.LayerNormalization(epsilon=1e-6, axis=-1)(cv2_f)
    cv3_f = layers.Dense(100, activation=None)(layers.Flatten()(cv3))
    #cv3_f = layers.LayerNormalization(epsilon=1e-6, axis=-1)(cv3_f)
    concat = layers.Concatenate(name='Concatenate1')([cv1_f, cv2_f, cv3_f])
    concat_d1 = layers.Dense(coor_dim*coor_dim, activation='relu')(concat)
    w1 = layers.Reshape(target_shape=(coor_dim, coor_dim))(concat_d1)

    cv1_f = layers.Dense(100, activation=None)(layers.Flatten()(cv1))
    # cv1_f = layers.LayerNormalization(epsilon=1e-6, axis=-1)(cv1_f)
    cv2_f = layers.Dense(100, activation=None)(layers.Flatten()(cv2))
    # cv2_f = layers.LayerNormalization(epsilon=1e-6, axis=-1)(cv2_f)
    cv3_f = layers.Dense(100, activation=None)(layers.Flatten()(cv3))
    # cv3_f = layers.LayerNormalization(epsilon=1e-6, axis=-1)(cv3_f)
    concat = layers.Concatenate(name='Concatenate2')([cv1_f, cv2_f, cv3_f])
    concat_d1 = layers.Dense(coor_dim * coor_dim, activation='relu')(concat)
    w2 = layers.Reshape(target_shape=(coor_dim, coor_dim))(concat_d1)

    initializer = tf.keras.initializers.GlorotUniform()  # Initialize weights from Glorot Uniform distribution
    w3 = tf.Variable(initializer(shape=(coor_dim, coor_dim)), trainable=True)  # (coor_dim, coor_dim)

    Q = tf.matmul(x_prime, w1)  # # (None, duration/4, coor_dim) @ (coor_dim, coor_dim) = (None, duration/4, coor_dim)
    K = tf.matmul(x_prime, w2) # # (None, duration/4, coor_dim) @ (coor_dim, coor_dim) = (None, duration/4, coor_dim)
    V = tf.matmul(x_prime, w3) # # (None, duration/4, coor_dim) @ (coor_dim, coor_dim) = (None, duration/4, coor_dim)

    scores = tf.matmul(Q, K, transpose_a=True) / tf.math.sqrt(tf.cast(coor_dim, tf.float32))  # (None, coor_dim, duration/4) @ (None, duration/4, coor_dim) = (None, coor_dim, coor_dim)
    flatten_scores = layers.Flatten()(scores)  # (None, coor_dim*coor_dim)
    softmax_scores = layers.Activation('softmax')(flatten_scores)  # (None, coor_dim*coor_dim)
    weights = layers.Reshape(target_shape=(scores.shape[1], scores.shape[2]), name='weights')(softmax_scores)  # (None, coor_dim, coor_dim)

    VW = tf.matmul(V, weights)  # (None, duration/4, coor_dim) @ (None, coor_dim, coor_dim) = (None, duration/4, coor_dim)
    # wx1 = layers.LayerNormalization(epsilon=1e-6, axis=[1, 2])(wx1)  # Normalize all weights in wx1
    # wx2 = layers.LayerNormalization(epsilon=1e-6, axis=[1, 2])(wx2) # Normalize all weights in wx2
    # wx3 = layers.LayerNormalization(epsilon=1e-6, axis=[1, 2])(wx3) # Normalize all weights in wx3

    f = layers.Flatten()(VW)
    drop = layers.Dropout(0.1)(f)
    d1 = layers.Dense(128, activation='relu')(drop)
    do1 = layers.Dropout(rate=0.1)(d1)
    d2 = layers.Dense(64, activation='relu')(do1)
    do2 = layers.Dropout(rate=0.1)(d2)
    d3 = layers.Dense(32, activation='relu', name='latent_vector')(do2)
    do3 = layers.Dropout(rate=0.1)(d3)
    d4 = layers.Dense(16, activation='relu')(do3)
    do4 = layers.Dropout(rate=0.1)(d4)

    y = layers.Dense(n_classes, activation='softmax')(do4)

    model = models.Model(x, y)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    model.summary()

    return model

def Res_Conv1D_LSTM_classifier(duration, coor_dim, n_classes):
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

    d1 = layers.Dense(128, activation='relu')(lstm)
    do1 = layers.Dropout(rate=0.2)(d1)
    d2 = layers.Dense(64, activation='relu')(do1)
    do2 = layers.Dropout(rate=0.2)(d2)
    d3 = layers.Dense(32, activation='relu')(do2)
    do3 = layers.Dropout(rate=0.1)(d3)
    d4 = layers.Dense(16, activation='relu')(do3)
    do4 = layers.Dropout(rate=0.1)(d4)

    if n_classes == 2:  # Binary classification
        y = layers.Dense(1, activation='sigmoid')(do4)
        model = models.Model(x, y)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
    elif n_classes >= 3:  # Multi-class classification
        y = layers.Dense(n_classes, activation='softmax')(do4)
        model = models.Model(x, y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    elif n_classes == 0:  # Regression
        y = layers.Dense(1, activation='linear')(do4)
        model = models.Model(x, y)
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      # metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
                      )

    #model.summary()

    return model


def scTRAIT_old(duration, embed_dim, n_classes):

    tf.keras.backend.clear_session()
    def causal_res_conv1d_block(input_layer, kernel_size, num_filters, dilation_rate):
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            input_layer)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
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
        conv = layers.Activation('leaky_relu')(conv)

        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('leaky_relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same', strides=1)(image)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    def spatial_extractor(x:tf.Tensor, kernel_size_2d:int, f_dim:int, name:str):
        ''' Extract spatial features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with spatial information (batch, f_dim)
        '''
        x_2d = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)  # (None, 32, 2, 1)
        cv1_2d = res_conv2d_block(x_2d, kernel_size=kernel_size_2d, num_filters=16, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 16)
        cv2_2d = res_conv2d_block(cv1_2d, kernel_size=kernel_size_2d, num_filters=32, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 32)
        f_2d = layers.Flatten()(cv2_2d)  # (None, 32*2*32)
        # drop_2d = layers.Dropout(0.1)(f_2d)  # (None, 32*2*32)
        y = layers.Dense(f_dim, activation=None)(f_2d)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def temporal_extractor(x:tf.Tensor, f_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with temporal information (batch, f_dim)
        '''
        cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)  # (None, 32, 16)
        p1 = layers.MaxPooling1D(pool_size=2)(cv1)  # (None, 16, 16)
        cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)  # (None, 16, 32)
        p2 = layers.MaxPooling1D(pool_size=2)(cv2)  # (None, 8, 32)
        cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)  # (None, 8, 64)
        f = layers.Flatten()(cv3)  # (None, 8*64)
        # drop = layers.Dropout(0.1)(f)  # (None, 8*64)
        y = layers.Dense(f_dim, activation=None)(f)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def spatiotemporal_composition(x:tf.Tensor, kernel_size_2d:int, f_dim:int, embed_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of each spatial and temporal latent feaures
        embed_dim: int
            Size of output latent features
        name: str
            Name of latent feaures
        Returns:
        -------
        model: models.Model
            Model that combines spatial features and temporal features and output combined latent feature (batch, embed_dim)
        '''

        spatial_f = spatial_extractor(x, kernel_size_2d=kernel_size_2d, f_dim=f_dim, name='spatial_f_%s'%name)  # (None, f_dim)
        temporal_f = temporal_extractor(x, f_dim=f_dim, name='temporal_f_%s'%name)  # (None, f_dim)
        y = layers.Concatenate(name='Concat_%s'%name)([spatial_f, temporal_f])  # (None, f_dim+f_dim)
        y = layers.Dense(embed_dim, activation=None)(y)  # (None, embed_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, embed_dim)
        y = layers.Activation('leaky_relu', name='emb_%s'%name)(y)  # (None, embed_dim)
        model = models.Model(inputs=x, outputs=y)

        return model

    def switch_layer(inputs):
        inp, emb = inputs
        zeros = tf.zeros_like(inp)  # tensor of zeros with same shape with inp
        ones = tf.ones_like(inp)

        inp = tf.keras.backend.switch(inp > 0, ones, zeros)  # Change inp>0 as 1, otherwise 0. (None, 1)
        inp = tf.cast(inp, tf.float32)  # dtype int -> float. (None, 1)
        inp = tf.expand_dims(inp, -1)  # (None, 1, 1)
        return inp * emb

    def delta_embedding_layer(x:tf.Tensor, embed_dim:int, name:str):
        ''' Get perturbation embeddings, where control = 0, else (emb_dim, ) vector
        Parameters:
        ----------
        x: tf.Tensor
            perturbation input (batch, 1)
        embed_dim: int
            Size of perturbation embedding
        name: str
            Name of latent feaures
        Returns:
        -------
        delta_emb: tf.Tensor
            Delta due to perturbation, where 0 in control. (batch, embed_dim)
        '''

        emb = layers.Embedding(4, embed_dim)(x)  # (None, 1, 64)
        #delta_emb = layers.Lambda(switch_layer)([x, emb])  # (None, 1, 64) Change control vector as 0
        delta_emb = layers.Reshape(target_shape=(emb.shape[-1],))(emb)  # (None, 64)
        delta_emb = layers.BatchNormalization(axis=-1)(delta_emb)  # (None, 64)
        delta_emb = layers.Activation('leaky_relu', name=name)(delta_emb)  # (None, 64)

        return delta_emb

    def MLP(x, embed_dim, drop_out, name):
        x = layers.Dense(embed_dim, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out > 0:
            x = layers.Dropout(rate=0.1)(x)
        y = layers.BatchNormalization(axis=-1, name=name)(x)
        return y

    x_traj = layers.Input(shape=(duration, 2))  # (None, 32, 2)
    traj_model = spatiotemporal_composition(x_traj, kernel_size_2d=2, f_dim=embed_dim, embed_dim=embed_dim, name='traj')  # (None, embed_dim)

    x_local = layers.Input(shape=(duration, 4))  # (None, 32, 4)
    local_model = spatiotemporal_composition(x_local, kernel_size_2d=3, f_dim=embed_dim, embed_dim=embed_dim,name='local')  # (None, embed_dim)

    # x_local = layers.Input(shape=(duration, 2))  # (None, 32, 2) if only
    # local_model = spatiotemporal_composition(x_local, kernel_size_2d=2, f_dim=128, embed_dim=embed_dim, name='local')  # (None, embed_dim)

    x_morph = layers.Input(shape=(duration, 8))  # (None, 32, 8)
    morph_model = spatiotemporal_composition(x_morph, kernel_size_2d=3, f_dim=embed_dim, embed_dim=embed_dim, name='morph')  # (None, embed_dim)


    x_covar = layers.Input(shape=(1,))  # (None, 1)
    delta_emb_traj = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='delta_emb_traj')  # (None, embed_dim)
    delta_emb_local = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='delta_emb_local')  # (None, embed_dim)
    delta_emb_morph = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='delta_emb_morph')  # (None, embed_dim)

    perturbed_traj = layers.add([traj_model.output, delta_emb_traj], name='perturbed_emb_traj')  # (None, embed_dim)
    perturbed_local = layers.add([local_model.output, delta_emb_local], name='perturbed_emb_local')  # (None, embed_dim)
    perturbed_morph = layers.add([morph_model.output, delta_emb_morph], name='perturbed_emb_morph')  # (None, embed_dim)

    traj_d = MLP(perturbed_traj, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_traj')  # (None, embed_dim//2)
    local_d = MLP(perturbed_local, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_local')  # (None, embed_dim//2)
    morph_d = MLP(perturbed_morph, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_morph')  # (None, embed_dim//2)

    concat = layers.Concatenate(name='Concat_all')([traj_d, local_d, morph_d])  # (None, 3*embed_dim//2)

    final = MLP(concat, embed_dim=embed_dim, drop_out=0.1, name='MLP_merged')
    final = layers.Dense(embed_dim // 2, activation='leaky_relu')(final)
    final = layers.Dense(embed_dim // 4, activation='leaky_relu')(final)

    if n_classes == 2:  # Binary classification
        y = layers.Dense(1, activation='sigmoid')(final)
        model = models.Model(inputs=[x_traj, x_local, x_morph, x_covar], outputs=y)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               #tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
    elif n_classes >= 3:  # Multi-class classification
        y = layers.Dense(n_classes, activation='softmax')(final)
        model = models.Model(inputs=[x_traj, x_local, x_morph, x_covar], outputs=y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    elif n_classes == 0:  # Regression
        y = layers.Dense(1, activation='linear')(final)
        model = models.Model(inputs=[x_traj, x_local, x_morph, x_covar], outputs=y)
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      #metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
                               )

    #model.summary()

    return model




def scTRAIT(duration, embed_dim, n_classes):

    tf.keras.backend.clear_session()
    def causal_res_conv1d_block(input_layer, kernel_size, num_filters, dilation_rate):
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            input_layer)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
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
        conv = layers.Activation('leaky_relu')(conv)

        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('leaky_relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same', strides=1)(image)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    def spatial_extractor(x:tf.Tensor, kernel_size_2d:int, f_dim:int, name:str):
        ''' Extract spatial features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with spatial information (batch, f_dim)
        '''
        x_2d = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)  # (None, 32, 2, 1)
        cv1_2d = res_conv2d_block(x_2d, kernel_size=kernel_size_2d, num_filters=16, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 16)
        cv2_2d = res_conv2d_block(cv1_2d, kernel_size=kernel_size_2d, num_filters=32, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 32)
        f_2d = layers.Flatten()(cv2_2d)  # (None, 32*2*32)
        # drop_2d = layers.Dropout(0.1)(f_2d)  # (None, 32*2*32)
        y = layers.Dense(f_dim, activation=None)(f_2d)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def temporal_extractor(x:tf.Tensor, f_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with temporal information (batch, f_dim)
        '''
        cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)  # (None, 32, 16)
        p1 = layers.MaxPooling1D(pool_size=2)(cv1)  # (None, 16, 16)
        cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)  # (None, 16, 32)
        p2 = layers.MaxPooling1D(pool_size=2)(cv2)  # (None, 8, 32)
        cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)  # (None, 8, 64)
        f = layers.Flatten()(cv3)  # (None, 8*64)
        # drop = layers.Dropout(0.1)(f)  # (None, 8*64)
        y = layers.Dense(f_dim, activation=None)(f)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def spatiotemporal_composition(x:tf.Tensor, kernel_size_2d:int, f_dim:int, embed_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of each spatial and temporal latent feaures
        embed_dim: int
            Size of output latent features
        name: str
            Name of latent feaures
        Returns:
        -------
        model: models.Model
            Model that combines spatial features and temporal features and output combined latent feature (batch, embed_dim)
        '''

        spatial_f = spatial_extractor(x, kernel_size_2d=kernel_size_2d, f_dim=f_dim, name='spatial_f_%s'%name)  # (None, f_dim)
        temporal_f = temporal_extractor(x, f_dim=f_dim, name='temporal_f_%s'%name)  # (None, f_dim)
        y = layers.Concatenate(name='Concat_%s'%name)([spatial_f, temporal_f])  # (None, f_dim+f_dim)
        y = layers.Dense(embed_dim, activation=None)(y)  # (None, embed_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, embed_dim)
        y = layers.Activation('leaky_relu', name='emb_%s'%name)(y)  # (None, embed_dim)
        model = models.Model(inputs=x, outputs=y)

        return model

    def switch_layer(inputs):
        inp, emb = inputs
        zeros = tf.zeros_like(inp)  # tensor of zeros with same shape with inp
        ones = tf.ones_like(inp)

        inp = tf.keras.backend.switch(inp > 0, ones, zeros)  # Change inp>0 as 1, otherwise 0. (None, 1)
        inp = tf.cast(inp, tf.float32)  # dtype int -> float. (None, 1)
        inp = tf.expand_dims(inp, -1)  # (None, 1, 1)
        return inp * emb

    def delta_embedding_layer(x:tf.Tensor, embed_dim:int, name:str):
        ''' Get perturbation embeddings, where control = 0, else (emb_dim, ) vector
        Parameters:
        ----------
        x: tf.Tensor
            perturbation input (batch, 1)
        embed_dim: int
            Size of perturbation embedding
        name: str
            Name of latent feaures
        Returns:
        -------
        delta_emb: tf.Tensor
            Delta due to perturbation, where 0 in control. (batch, embed_dim)
        '''

        emb = layers.Embedding(4, embed_dim)(x)  # (None, 1, 64)
        #delta_emb = layers.Lambda(switch_layer)([x, emb])  # (None, 1, 64) Change control vector as 0
        delta_emb = layers.Reshape(target_shape=(emb.shape[-1],))(emb)  # (None, 64)
        delta_emb = layers.BatchNormalization(axis=-1)(delta_emb)  # (None, 64)
        delta_emb = layers.Activation('leaky_relu', name=name)(delta_emb)  # (None, 64)

        return delta_emb

    def MLP(x, embed_dim, drop_out, name):
        x = layers.Dense(embed_dim, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out > 0:
            x = layers.Dropout(rate=0.1)(x)
        y = layers.BatchNormalization(axis=-1, name=name)(x)
        return y

    x_traj = layers.Input(shape=(duration, 2))  # (None, 32, 2)
    traj_model = spatiotemporal_composition(x_traj, kernel_size_2d=2, f_dim=embed_dim, embed_dim=embed_dim, name='traj')  # (None, embed_dim)

    x_local = layers.Input(shape=(duration, 4))  # (None, 32, 4)
    local_model = spatiotemporal_composition(x_local, kernel_size_2d=3, f_dim=embed_dim, embed_dim=embed_dim,name='local')  # (None, embed_dim)

    # x_local = layers.Input(shape=(duration, 2))  # (None, 32, 2) if only
    # local_model = spatiotemporal_composition(x_local, kernel_size_2d=2, f_dim=128, embed_dim=embed_dim, name='local')  # (None, embed_dim)

    x_morph = layers.Input(shape=(duration, 8))  # (None, 32, 8)
    morph_model = spatiotemporal_composition(x_morph, kernel_size_2d=3, f_dim=embed_dim, embed_dim=embed_dim, name='morph')  # (None, embed_dim)

    ############# Behavior-specific delta layer #############
    x_covar = layers.Input(shape=(1,))  # (None, 1)
    init_delta_emb_traj = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='initial_delta_emb_traj')  # (None, embed_dim)
    init_delta_emb_local = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='initial_delta_emb_local')  # (None, embed_dim)
    init_delta_emb_morph = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='initial_delta_emb_morph')  # (None, embed_dim)

    concat_traj = layers.Concatenate(name='concat_traj')([traj_model.output, init_delta_emb_traj])  # (None, 2*embed_dim)
    concat_local = layers.Concatenate(name='concat_local')([local_model.output, init_delta_emb_local])  # (None, 2*embed_dim)
    concat_morph = layers.Concatenate(name='concat_morph')([morph_model.output, init_delta_emb_morph])  # (None, 2*embed_dim)

    delta_emb_traj = MLP(concat_traj, embed_dim=2 * embed_dim, drop_out=0.1, name='delta_emb_traj')  # (None, embed_dim)
    delta_emb_local = MLP(concat_local, embed_dim=2 * embed_dim, drop_out=0.1, name='delta_emb_local')  # (None, embed_dim)
    delta_emb_morph = MLP(concat_morph, embed_dim=2 * embed_dim, drop_out=0.1, name='delta_emb_morph')  # (None, embed_dim)

    ############# Add perturbation #############
    perturbed_traj = layers.add([traj_model.output, delta_emb_traj], name='perturbed_emb_traj')  # (None, embed_dim)
    perturbed_local = layers.add([local_model.output, delta_emb_local], name='perturbed_emb_local')  # (None, embed_dim)
    perturbed_morph = layers.add([morph_model.output, delta_emb_morph], name='perturbed_emb_morph')  # (None, embed_dim)

    traj_d = MLP(perturbed_traj, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_traj')  # (None, embed_dim//2)
    local_d = MLP(perturbed_local, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_local')  # (None, embed_dim//2)
    morph_d = MLP(perturbed_morph, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_morph')  # (None, embed_dim//2)

    concat = layers.Concatenate(name='concat_all')([traj_d, local_d, morph_d])  # (None, 3*embed_dim//2)

    final = MLP(concat, embed_dim=embed_dim, drop_out=0.1, name='MLP_merged')
    final = layers.Dense(embed_dim // 2, activation='leaky_relu')(final)
    final = layers.Dense(embed_dim // 4, activation='leaky_relu')(final)

    if n_classes == 2:  # Binary classification
        y = layers.Dense(1, activation='sigmoid')(final)
        model = models.Model(inputs=[x_traj, x_local, x_morph, x_covar], outputs=y)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               #tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
    elif n_classes >= 3:  # Multi-class classification
        y = layers.Dense(n_classes, activation='softmax')(final)
        model = models.Model(inputs=[x_traj, x_local, x_morph, x_covar], outputs=y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    elif n_classes == 0:  # Regression
        y = layers.Dense(1, activation='linear')(final)
        model = models.Model(inputs=[x_traj, x_local, x_morph, x_covar], outputs=y)
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      #metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
                               )

    #model.summary()

    return model


def scTRAIT_remove_morph(duration, embed_dim, n_classes):

    tf.keras.backend.clear_session()
    def causal_res_conv1d_block(input_layer, kernel_size, num_filters, dilation_rate):
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            input_layer)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
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
        conv = layers.Activation('leaky_relu')(conv)

        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('leaky_relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same', strides=1)(image)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    def spatial_extractor(x:tf.Tensor, kernel_size_2d:int, f_dim:int, name:str):
        ''' Extract spatial features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with spatial information (batch, f_dim)
        '''
        x_2d = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)  # (None, 32, 2, 1)
        cv1_2d = res_conv2d_block(x_2d, kernel_size=kernel_size_2d, num_filters=16, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 16)
        cv2_2d = res_conv2d_block(cv1_2d, kernel_size=kernel_size_2d, num_filters=32, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 32)
        f_2d = layers.Flatten()(cv2_2d)  # (None, 32*2*32)
        # drop_2d = layers.Dropout(0.1)(f_2d)  # (None, 32*2*32)
        y = layers.Dense(f_dim, activation=None)(f_2d)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def temporal_extractor(x:tf.Tensor, f_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with temporal information (batch, f_dim)
        '''
        cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)  # (None, 32, 16)
        p1 = layers.MaxPooling1D(pool_size=2)(cv1)  # (None, 16, 16)
        cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)  # (None, 16, 32)
        p2 = layers.MaxPooling1D(pool_size=2)(cv2)  # (None, 8, 32)
        cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)  # (None, 8, 64)
        f = layers.Flatten()(cv3)  # (None, 8*64)
        # drop = layers.Dropout(0.1)(f)  # (None, 8*64)
        y = layers.Dense(f_dim, activation=None)(f)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def spatiotemporal_composition(x:tf.Tensor, kernel_size_2d:int, f_dim:int, embed_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of each spatial and temporal latent feaures
        embed_dim: int
            Size of output latent features
        name: str
            Name of latent feaures
        Returns:
        -------
        model: models.Model
            Model that combines spatial features and temporal features and output combined latent feature (batch, embed_dim)
        '''

        spatial_f = spatial_extractor(x, kernel_size_2d=kernel_size_2d, f_dim=f_dim, name='spatial_f_%s'%name)  # (None, f_dim)
        temporal_f = temporal_extractor(x, f_dim=f_dim, name='temporal_f_%s'%name)  # (None, f_dim)
        y = layers.Concatenate(name='Concat_%s'%name)([spatial_f, temporal_f])  # (None, f_dim+f_dim)
        y = layers.Dense(embed_dim, activation=None)(y)  # (None, embed_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, embed_dim)
        y = layers.Activation('leaky_relu', name='emb_%s'%name)(y)  # (None, embed_dim)
        model = models.Model(inputs=x, outputs=y)

        return model

    def switch_layer(inputs):
        inp, emb = inputs
        zeros = tf.zeros_like(inp)  # tensor of zeros with same shape with inp
        ones = tf.ones_like(inp)

        inp = tf.keras.backend.switch(inp > 0, ones, zeros)  # Change inp>0 as 1, otherwise 0. (None, 1)
        inp = tf.cast(inp, tf.float32)  # dtype int -> float. (None, 1)
        inp = tf.expand_dims(inp, -1)  # (None, 1, 1)
        return inp * emb

    def delta_embedding_layer(x:tf.Tensor, embed_dim:int, name:str):
        ''' Get perturbation embeddings, where control = 0, else (emb_dim, ) vector
        Parameters:
        ----------
        x: tf.Tensor
            perturbation input (batch, 1)
        embed_dim: int
            Size of perturbation embedding
        name: str
            Name of latent feaures
        Returns:
        -------
        delta_emb: tf.Tensor
            Delta due to perturbation, where 0 in control. (batch, embed_dim)
        '''

        emb = layers.Embedding(4, embed_dim)(x)  # (None, 1, 64)
        #delta_emb = layers.Lambda(switch_layer)([x, emb])  # (None, 1, 64) Change control vector as 0
        delta_emb = layers.Reshape(target_shape=(emb.shape[-1],))(emb)  # (None, 64)
        delta_emb = layers.BatchNormalization(axis=-1)(delta_emb)  # (None, 64)
        delta_emb = layers.Activation('leaky_relu', name=name)(delta_emb)  # (None, 64)

        return delta_emb

    def MLP(x, embed_dim, drop_out, name):
        x = layers.Dense(embed_dim, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out > 0:
            x = layers.Dropout(rate=0.1)(x)
        y = layers.BatchNormalization(axis=-1, name=name)(x)
        return y

    x_traj = layers.Input(shape=(duration, 2))  # (None, 32, 2)
    traj_model = spatiotemporal_composition(x_traj, kernel_size_2d=2, f_dim=embed_dim, embed_dim=embed_dim, name='traj')  # (None, embed_dim)

    x_local = layers.Input(shape=(duration, 4))  # (None, 32, 4)
    local_model = spatiotemporal_composition(x_local, kernel_size_2d=3, f_dim=embed_dim, embed_dim=embed_dim,name='local')  # (None, embed_dim)

    # x_local = layers.Input(shape=(duration, 2))  # (None, 32, 2) if only
    # local_model = spatiotemporal_composition(x_local, kernel_size_2d=2, f_dim=128, embed_dim=embed_dim, name='local')  # (None, embed_dim)


    ############# Behavior-specific delta layer #############
    x_covar = layers.Input(shape=(1,))  # (None, 1)
    init_delta_emb_traj = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='initial_delta_emb_traj')  # (None, embed_dim)
    init_delta_emb_local = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='initial_delta_emb_local')  # (None, embed_dim)

    concat_traj = layers.Concatenate(name='concat_traj')([traj_model.output, init_delta_emb_traj])  # (None, 2*embed_dim)
    concat_local = layers.Concatenate(name='concat_local')([local_model.output, init_delta_emb_local])  # (None, 2*embed_dim)

    delta_emb_traj = MLP(concat_traj, embed_dim=2 * embed_dim, drop_out=0.1, name='delta_emb_traj')  # (None, embed_dim)
    delta_emb_local = MLP(concat_local, embed_dim=2 * embed_dim, drop_out=0.1, name='delta_emb_local')  # (None, embed_dim)

    ############# Add perturbation #############
    perturbed_traj = layers.add([traj_model.output, delta_emb_traj], name='perturbed_emb_traj')  # (None, embed_dim)
    perturbed_local = layers.add([local_model.output, delta_emb_local], name='perturbed_emb_local')  # (None, embed_dim)

    traj_d = MLP(perturbed_traj, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_traj')  # (None, embed_dim//2)
    local_d = MLP(perturbed_local, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_local')  # (None, embed_dim//2)

    concat = layers.Concatenate(name='concat_all')([traj_d, local_d])  # (None, embed_dim)

    final = MLP(concat, embed_dim=embed_dim, drop_out=0.1, name='MLP_merged')  # (None, embed_dim//4)
    final = layers.Dense(embed_dim // 4, activation='leaky_relu')(final)  # (None, embed_dim//4)
    final = layers.Dense(embed_dim // 8, activation='leaky_relu')(final)  # (None, embed_dim//8)

    if n_classes == 2:  # Binary classification
        y = layers.Dense(1, activation='sigmoid')(final)
        model = models.Model(inputs=[x_traj, x_local, x_covar], outputs=y)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               #tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
    elif n_classes >= 3:  # Multi-class classification
        y = layers.Dense(n_classes, activation='softmax')(final)
        model = models.Model(inputs=[x_traj, x_local, x_covar], outputs=y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    elif n_classes == 0:  # Regression
        y = layers.Dense(1, activation='linear')(final)
        model = models.Model(inputs=[x_traj, x_local, x_covar], outputs=y)
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      #metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
                               )

    #model.summary()

    return model



def scTRAIT_only_traj(duration, embed_dim, n_classes):

    tf.keras.backend.clear_session()
    def causal_res_conv1d_block(input_layer, kernel_size, num_filters, dilation_rate):
        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            input_layer)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(
            x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation('leaky_relu')(x)
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
        conv = layers.Activation('leaky_relu')(conv)

        conv = layers.Conv2D(num_filters, (kernel_size, kernel_size), padding='same', strides=1)(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=-1)(
                conv)  # input = (num_of_images, height, width, channels)에서 channels 방향으로 normalize
        conv = layers.Activation('leaky_relu')(conv)

        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        shortcut = layers.Conv2D(num_filters, (1, 1), padding='same', strides=1)(image)
        if batch_norm is True:
            shortcut = layers.BatchNormalization(axis=-1)(shortcut)

        residual = layers.add([shortcut, conv])

        return residual

    def spatial_extractor(x:tf.Tensor, kernel_size_2d:int, f_dim:int, name:str):
        ''' Extract spatial features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with spatial information (batch, f_dim)
        '''
        x_2d = layers.Reshape(target_shape=(x.shape[1], x.shape[2], 1))(x)  # (None, 32, 2, 1)
        cv1_2d = res_conv2d_block(x_2d, kernel_size=kernel_size_2d, num_filters=16, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 16)
        cv2_2d = res_conv2d_block(cv1_2d, kernel_size=kernel_size_2d, num_filters=32, dropout=0.1,
                                  batch_norm=True)  # (None, 32, 2, 32)
        f_2d = layers.Flatten()(cv2_2d)  # (None, 32*2*32)
        # drop_2d = layers.Dropout(0.1)(f_2d)  # (None, 32*2*32)
        y = layers.Dense(f_dim, activation=None)(f_2d)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def temporal_extractor(x:tf.Tensor, f_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        f_dim: int
            Size of latent feaures
        name: str
            Name of latent feaures
        Returns:
        -------
        y: tf.Tensor
            latent features associated with temporal information (batch, f_dim)
        '''
        cv1 = causal_res_conv1d_block(x, kernel_size=3, num_filters=16, dilation_rate=1)  # (None, 32, 16)
        p1 = layers.MaxPooling1D(pool_size=2)(cv1)  # (None, 16, 16)
        cv2 = causal_res_conv1d_block(p1, kernel_size=3, num_filters=32, dilation_rate=2)  # (None, 16, 32)
        p2 = layers.MaxPooling1D(pool_size=2)(cv2)  # (None, 8, 32)
        cv3 = causal_res_conv1d_block(p2, kernel_size=3, num_filters=64, dilation_rate=4)  # (None, 8, 64)
        f = layers.Flatten()(cv3)  # (None, 8*64)
        # drop = layers.Dropout(0.1)(f)  # (None, 8*64)
        y = layers.Dense(f_dim, activation=None)(f)  # (None, f_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, f_dim)
        y = layers.Activation('leaky_relu', name=name)(y)  # (None, f_dim)

        return y

    def spatiotemporal_composition(x:tf.Tensor, kernel_size_2d:int, f_dim:int, embed_dim:int, name:str):
        ''' Extract temporal features from timeseries
        Parameters:
        ----------
        x: tf.Tensor
            input Tensor (batch, duration, 2) if 2-dimensional timeseries
        kernel_size_2d: int
            apply (kernel_size_2d, kernel_size_2d) kerenl for 2D convolution
        f_dim: int
            Size of each spatial and temporal latent feaures
        embed_dim: int
            Size of output latent features
        name: str
            Name of latent feaures
        Returns:
        -------
        model: models.Model
            Model that combines spatial features and temporal features and output combined latent feature (batch, embed_dim)
        '''

        spatial_f = spatial_extractor(x, kernel_size_2d=kernel_size_2d, f_dim=f_dim, name='spatial_f_%s'%name)  # (None, f_dim)
        temporal_f = temporal_extractor(x, f_dim=f_dim, name='temporal_f_%s'%name)  # (None, f_dim)
        y = layers.Concatenate(name='Concat_%s'%name)([spatial_f, temporal_f])  # (None, f_dim+f_dim)
        y = layers.Dense(embed_dim, activation=None)(y)  # (None, embed_dim)
        y = layers.BatchNormalization(axis=-1)(y)  # (None, embed_dim)
        y = layers.Activation('leaky_relu', name='emb_%s'%name)(y)  # (None, embed_dim)
        model = models.Model(inputs=x, outputs=y)

        return model

    def switch_layer(inputs):
        inp, emb = inputs
        zeros = tf.zeros_like(inp)  # tensor of zeros with same shape with inp
        ones = tf.ones_like(inp)

        inp = tf.keras.backend.switch(inp > 0, ones, zeros)  # Change inp>0 as 1, otherwise 0. (None, 1)
        inp = tf.cast(inp, tf.float32)  # dtype int -> float. (None, 1)
        inp = tf.expand_dims(inp, -1)  # (None, 1, 1)
        return inp * emb

    def delta_embedding_layer(x:tf.Tensor, embed_dim:int, name:str):
        ''' Get perturbation embeddings, where control = 0, else (emb_dim, ) vector
        Parameters:
        ----------
        x: tf.Tensor
            perturbation input (batch, 1)
        embed_dim: int
            Size of perturbation embedding
        name: str
            Name of latent feaures
        Returns:
        -------
        delta_emb: tf.Tensor
            Delta due to perturbation, where 0 in control. (batch, embed_dim)
        '''

        emb = layers.Embedding(2, embed_dim)(x)  # (None, 1, 64)
        #delta_emb = layers.Lambda(switch_layer)([x, emb])  # (None, 1, 64) Change control vector as 0
        delta_emb = layers.Reshape(target_shape=(emb.shape[-1],))(emb)  # (None, 64)
        delta_emb = layers.BatchNormalization(axis=-1)(delta_emb)  # (None, 64)
        delta_emb = layers.Activation('leaky_relu', name=name)(delta_emb)  # (None, 64)

        return delta_emb

    def MLP(x, embed_dim, drop_out, name):
        x = layers.Dense(embed_dim, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out>0:
            x = layers.Dropout(rate=0.1)(x)
        x = layers.Dense(embed_dim // 2, activation='leaky_relu')(x)
        if drop_out > 0:
            x = layers.Dropout(rate=0.1)(x)
        y = layers.BatchNormalization(axis=-1, name=name)(x)
        return y

    x_traj = layers.Input(shape=(duration, 2))  # (None, 32, 2)
    traj_model = spatiotemporal_composition(x_traj, kernel_size_2d=2, f_dim=embed_dim, embed_dim=embed_dim, name='traj')  # (None, embed_dim)

    ############# Behavior-specific delta layer #############
    x_covar = layers.Input(shape=(1,))  # (None, 1)
    init_delta_emb_traj = delta_embedding_layer(x_covar, embed_dim=embed_dim, name='initial_delta_emb_traj')  # (None, embed_dim)
    concat_traj = layers.Concatenate(name='concat_traj')([traj_model.output, init_delta_emb_traj])  # (None, 2*embed_dim)
    delta_emb_traj = MLP(concat_traj, embed_dim=2 * embed_dim, drop_out=0.1, name='delta_emb_traj')  # (None, embed_dim)

    ############# Add perturbation #############
    perturbed_traj = layers.add([traj_model.output, delta_emb_traj], name='perturbed_emb_traj')  # (None, embed_dim)
    traj_d = MLP(perturbed_traj, embed_dim=embed_dim, drop_out=0.1, name='MLP_perturbed_emb_traj')  # (None, embed_dim//2)
    final = layers.Dense(embed_dim // 4, activation='leaky_relu')(traj_d)  # (None, embed_dim//4)
    final = layers.Dense(embed_dim // 8, activation='leaky_relu')(final)  # (None, embed_dim//8)

    if n_classes == 2:  # Binary classification
        y = layers.Dense(1, activation='sigmoid')(final)
        model = models.Model(inputs=[x_traj, x_covar], outputs=y)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                               #tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall')])
    elif n_classes >= 3:  # Multi-class classification
        y = layers.Dense(n_classes, activation='softmax')(final)
        model = models.Model(inputs=[x_traj, x_covar], outputs=y)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    elif n_classes == 0:  # Regression
        y = layers.Dense(1, activation='linear')(final)
        model = models.Model(inputs=[x_traj, x_covar], outputs=y)
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      #metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),]
                               )

    #model.summary()

    return model







