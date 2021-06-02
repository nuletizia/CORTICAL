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

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true* y_pred)

def my_binary_crossentropy(y_true, y_pred):
    return -K.mean(K.log(y_true)+K.log(y_pred))

class dDIME():
    def __init__(self, alpha, EbN0):

        # Input shape
        self.latent_dim = 2
        # Joint architecture
        self.joint_dim = 2 * self.latent_dim

        self.alpha = alpha

        # Noise std based on EbN0 in dB
        eps = np.sqrt(pow(10, -0.1 * EbN0) / (2 * 0.5))

        self.eps = eps

        optimizer = Adam(0.002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        xy = Input(shape=(self.joint_dim,))
        x_y = Input(shape=(self.joint_dim,))

        # The discriminator takes as input joint or marginal vectors
        d_xy = self.discriminator(xy)
        d_x_y = self.discriminator(x_y)

        # Train the discriminator

        self.combined = Model([xy, x_y], [d_xy,d_x_y])
        self.combined.compile(loss=[my_binary_crossentropy,wasserstein_loss],loss_weights=[self.alpha,1], optimizer=optimizer)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=self.joint_dim))
        model.add(GaussianNoise(0.3))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation='softplus'))

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

            d_loss = self.combined.train_on_batch([data_xy,data_x_y],[valid,valid])

            # Estimate the density ratio and the MI in two ways, hat and tilde (see paper)
            D_value = self.discriminator.predict(data_xy)
            R = (D_value)/self.alpha

            MI = np.log(R) # Print it during training if desired

            # Plot the progress
            print ("%d [D total loss : %f, D log loss : %f, D mean loss: %f" % (epoch, d_loss[0],d_loss[1], d_loss[2]))


    def test(self, test_size=10000):

        noise_real = np.random.normal(0, 1, (test_size, self.latent_dim))
        eps = self.eps

        noise_fake = noise_real + eps * np.random.normal(0, 1, (test_size, self.latent_dim))
        data_xy, data_x_y = data_generation_mi(noise_real, noise_fake)

        D_value_1 = self.discriminator.predict(data_xy)
        D_value_2 = self.discriminator.predict(data_x_y)

        # for tilde dDIME, variational lower bound, you need to average over the joint samples and the marginals
        J_e = self.alpha*np.log(D_value_1)-D_value_2
        MI_e = J_e/self.alpha+1-np.log(self.alpha)

        # for hat dDIME you need to average over the joint samples
        R = D_value_1/self.alpha
        MI_s = np.log(R)

        print(f'The mean value of tilde I is :{np.mean(MI_e)}')
        print(f'The mean value of hat I is :{np.mean(MI_s)}')

        return MI_e, MI_s

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='Number of data samples to train on at once', default=512)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=5000)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=10000)
    parser.add_argument('--alpha', help='Alpha parameter of dDIME', default=1)

    args = parser.parse_args()

    test_size = int(args.test_size)
    alpha = float(args.alpha)

    SNR_dB = range(-14, 29)
    MI_s_total = np.zeros((len(SNR_dB), test_size))
    MI_e_total = np.zeros((len(SNR_dB),test_size))

    j = 0
    for SNR in SNR_dB:
        print(f'Actual SNR is:{SNR}')
        # Initialize dDIME
        ddime = dDIME(alpha, SNR)
        # Train
        ddime.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
        # Test
        MI_e, MI_s = ddime.test(test_size=10000)
        MI_e_total[j,:] = np.transpose(MI_e)
        MI_s_total[j,:] = np.transpose(MI_s)
        del ddime
        j = j+1

    sio.savemat('data_dDMIE.mat', {'SNR': SNR_dB, 'MI_e': MI_e_total, 'MI_s': MI_s_total})
