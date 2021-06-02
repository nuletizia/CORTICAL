from __future__ import print_function, division

from keras.layers import Input, Dense, GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K

K.clear_session()
import numpy as np
import scipy.io as sio
import argparse

def data_generation_mi(data_x, data_y):
    data_xy = np.hstack((data_x, data_y))
    data_y_shuffle = np.take(data_y, np.random.permutation(data_y.shape[0]), axis=0, out=data_y)
    data_x_y = np.hstack((data_x, data_y_shuffle))
    return data_xy, data_x_y

class iDIME():
    def __init__(self, EbN0):

        # Input shape
        self.latent_dim = 2
        # Joint architecture
        self.joint_dim = 2*self.latent_dim
        # Noise std based on EbN0 in dB
        eps = np.sqrt(pow(10, -0.1 * EbN0) / (2 * 0.5))

        self.eps = eps

        optimizer = Adam(0.002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        z = Input(shape=(self.joint_dim,))

        # The discriminator takes as input joint or marginal vectors
        valid_1 = self.discriminator(z)

        # Train the discriminator
        self.combined = Model(z, valid_1)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=self.joint_dim))
        model.add(GaussianNoise(0.3))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        T = Input(shape=(self.joint_dim,))
        D = model(T)

        return Model(T, D)


    def train(self, epochs, batch_size=40):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate a batch
            noise_real = np.random.normal(0, 1, (batch_size, self.latent_dim))
            eps = self.eps
            noise_fake = noise_real + eps*np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Concatenate and shuffle, see pi() in the paper
            data_xy, data_x_y = data_generation_mi(noise_real, noise_fake)

            d_loss_real = self.discriminator.train_on_batch(data_xy, fake)
            d_loss_fake = self.discriminator.train_on_batch(data_x_y, valid)
            d_loss = np.add(d_loss_real, d_loss_fake)

            D_value = self.discriminator.predict(data_xy)

            # Estimate the density ratio and the MI
            R = (1-D_value)/(D_value)
            MI = np.log(R) # Print it during training if desired

            # Plot the progress
            print ("%d [D1 loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 50*d_loss[1]))


    def test(self, test_size=10000):

        # noise_real = 2 * np.random.randint(2, size=(test_size, self.latent_dim)) - 1 if you want balanced bernoulli, discrete case..
        noise_real = np.random.normal(0, 1, (test_size, self.latent_dim))
        eps = self.eps

        noise_fake = noise_real + eps * np.random.normal(0, 1, (test_size, self.latent_dim))
        data_xy, data_x_y = data_generation_mi(noise_real, noise_fake)

        # for iDIME you need to average over the joint samples
        D_value = self.discriminator.predict(data_xy)
        R = (1 - D_value)/D_value
        MI_s = np.log(R)

        print(f'The mean value of hat I is :{np.mean(MI_s)}')

        return MI_s

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Number of data samples to train on at once', default=512)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=5000)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=10000)

    args = parser.parse_args()

    test_size = int(args.test_size)

    SNR_dB = range(-14, 29)
    MI_s_total = np.zeros((len(SNR_dB),test_size))
    j = 0
    for SNR in SNR_dB:
        print(f'Actual SNR is:{SNR}')
        # Initialize iDIME
        idime = iDIME(SNR)
        # Train
        idime.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
        # Test
        MI_s = idime.test(test_size=10000)
        MI_s_total[j,:] = np.transpose(MI_s)
        del idime
        j = j+1

    sio.savemat('data_iDMIE.mat', {'SNR': SNR_dB, 'MI_s': MI_s_total})
