from __future__ import print_function, division

from keras.layers import BatchNormalization, Input, Dense, GaussianNoise, Concatenate, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
K.clear_session()
import tensorflow as tf
import numpy as np
import scipy.io as sio
import argparse

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from dDIME import dDIME, data_generation_mi


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true* y_pred)

def my_binary_crossentropy(y_true, y_pred):
    return -K.mean(K.log(y_true)+K.log(y_pred))

def shuffleColumns(x):
    #x = tf.transpose(x)
    #x = tf.random.shuffle(x)
    return tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0])))

class CORTICAL():
    def __init__(self, latent_dim, data_dim, alpha, EbN0):
        # Input shape of z
        self.latent_dim = latent_dim

        # Shape of channel input x
        self.data_dim = data_dim

        # Joint architecture shape
        self.joint_dim = 2 * self.data_dim
        self.alpha = alpha

        self.eps = np.sqrt(pow(10, -0.1 * EbN0) / (2 * 0.5))

        # Noise power
        N = self.eps**2

        optimizer_G = Adam(0.0002, 0.5)
        optimizer_D = Adam(0.002, 0.5)

        # Build and compile the discriminator
        self.discriminator = dDIME.build_discriminator(self)

        # Build the generators
        self.generator = self.build_generator()

        # The transmitter encodes the bits in s
        s_in = Input(shape=(self.latent_dim,))
        x = self.generator(s_in)

        # Build the channel
        # x_n = Lambda(lambda x: np.sqrt(N)*K.l2_normalize(x,axis=1))(x) #if CONSTANT POWER
        x_n = BatchNormalization(axis=-1, center=False, scale=False)(x)  # if AVERAGE POWER
        self.encoder = Model(s_in, x_n)

        ch = Lambda(lambda x: x)(x_n)

        y = GaussianNoise(np.sqrt(N))(ch)  # AWGN layer

        xy = Concatenate(name='network/concatenate_layer_1')([x_n, y])
        y_bar_input = Lambda(lambda x: shuffleColumns(x))(y)  # shuffle y input as y_bar
        x_y = Concatenate(name='network/concatenate_layer_2')([x_n, y_bar_input])

        # The discriminator takes as input joint or marginal vectors
        d_xy = self.discriminator(xy)
        d_x_y = self.discriminator(x_y)

        xy_in = Input(shape=(self.joint_dim,))
        x_y_in = Input(shape=(self.joint_dim,))

        d_xy_in = self.discriminator(xy_in)
        d_x_y_in = self.discriminator(x_y_in)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to help the discriminator

        self.combined_d = Model([xy_in,x_y_in], [d_xy_in,d_x_y_in])
        self.combined_d.compile(loss=[my_binary_crossentropy,wasserstein_loss],loss_weights=[self.alpha,1], optimizer=optimizer_D)

        # Update only the generator
        self.discriminator.trainable = False

        self.combined_g = Model(s_in, [d_xy, d_x_y])
        self.combined_g.compile(loss=[my_binary_crossentropy, wasserstein_loss], loss_weights=[self.alpha, 1],
                                optimizer=optimizer_G)


    def build_generator(self):

        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=self.latent_dim))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(self.data_dim))

        model.summary()

        T = Input(shape=(self.latent_dim,))
        D = model(T)

        return Model(T, D)


    def train(self, epochs, batch_size=40):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        epochs_D = 10
        epochs_G = 1

        for epoch in range(epochs):

            # Sample noise or bits and train CORTICAL

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for epoch_D in range(epochs_D):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                #noise = 2 * np.random.randint(2, size=(batch_size, self.latent_dim)) - 1, use only with discrete inputs
                data_real = self.encoder.predict(noise)
                eps = self.eps
                data_fake = data_real + eps * np.random.normal(0, 1, (batch_size, self.data_dim))
                data_xy, data_x_y = data_generation_mi(data_real, data_fake)

                d_loss = self.combined_d.train_on_batch([data_xy, data_x_y],[valid,valid])

            # Print these values if desired
            D_value = self.discriminator.predict(data_xy)
            R = (D_value) / self.alpha
            MI = np.log(R)

            # ---------------------
            #  Train Generator
            # ---------------------

            for epoch_G in range(epochs_G):
                #noise = 2 * np.random.randint(2, size=(batch_size, self.latent_dim)) - 1
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.combined_g.train_on_batch(noise, [valid, valid])

            # Plot the progress
            print("%d [D1 loss: %f] [D2 loss: %f] [G1 loss: %f]" % (epoch, d_loss[0],d_loss[1], g_loss[0]))

    def test(self, test_size=1000):

        # test now..
        noise = np.random.normal(0,1, (test_size, self.latent_dim))
        #noise = 2 * np.random.randint(2, size = (test_size, self.latent_dim)) - 1

        # Get channel input data
        data_real = self.encoder.predict(noise)

        eps = self.eps

        # Implement the AWGN channel
        data_fake = data_real + eps * np.random.normal(0, 1, (test_size, self.data_dim))

        # Shuffle to implement pi(), see paper
        data_xy, data_x_y = data_generation_mi(data_real, data_fake)

        # for tilde dDIME, variational lower bound, you need to average over the joint samples and the marginals
        D_value_1 = self.discriminator.predict(data_xy)
        D_value_2 = self.discriminator.predict(data_x_y)
        J_e = self.alpha * np.log(D_value_1) - D_value_2
        MI_e = J_e / self.alpha + 1 - np.log(self.alpha)

        # for hat dDIME you need to average over the joint samples
        R = (D_value_1) / self.alpha
        MI_s = np.log(R)

        # Return also the constellation
        return MI_e, MI_s, data_real, data_fake

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', help='Number of images to train on at once', default=512)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=5000)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=10000)
    parser.add_argument('--alpha', help='Alpha parameter of dDIME', default=1)

    args = parser.parse_args()

    test_size = int(args.test_size)
    alpha = float(args.alpha)

    EbN0_dB = range(-14, 29)

    latent_dim = 30 # z input dimension
    data_dim = 2 # x channel input dimension

    MI_e_total = np.zeros((len(EbN0_dB),test_size))
    MI_s_total = np.zeros((len(EbN0_dB), test_size))

    features_x = []
    features_y = []

    j = 0
    for EbN0 in EbN0_dB:
        print(f'Current EbN0 is:{EbN0}')
        # Initialize CORTICAL
        cortical = CORTICAL(latent_dim, data_dim, alpha, EbN0)
        # Train
        cortical.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
        # Test
        MI_e, MI_s, data_x, data_y = cortical.test(test_size)
        MI_e_total[j, :] = np.transpose(MI_e)
        MI_s_total[j, :] = np.transpose(MI_s)
        features_x.append(data_x)
        features_y.append(data_y)
        del cortical
        j = j + 1

    sio.savemat('data_CORTICAL.mat', {'MI_e': MI_e_total, 'MI_s': MI_s_total, 'features_x': features_x, 'features_y':features_y})