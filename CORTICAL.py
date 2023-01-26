from __future__ import print_function, division

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import BatchNormalization, Input, Dense, GaussianNoise, Concatenate, Lambda, regularizers
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K


import numpy as np
import scipy.io as sio
import argparse

from numpy.random import seed

from uniform_noise import UniformNoise
from cauchy_noise import CauchyNoise
from exponential_noise import ExponentialNoise

# loss functions and regularizations
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true* y_pred)

def my_binary_crossentropy(y_true, y_pred):
    return -K.mean(K.log(y_true)+K.log(y_pred))

def my_peak_power(y_true, y_pred):
    # P = A^2
    return K.maximum(K.sum(y_pred*y_pred,axis=-1,keepdims=True)-1,0)

def my_peak_power_ellipse(y_true, y_pred):
    # P = x^2+R*y^2
    R = 3
    return K.maximum(y_pred[:,0]*y_pred[:,0]+R*y_pred[:,1]*y_pred[:,1]-1,0)

def my_average_power(y_true, y_pred):
    return K.maximum(K.mean(K.sum(y_pred*y_pred,axis=-1,keepdims=True))-1,0)

def my_log_power(y_true, y_pred):
    A = 2
    gamma = 1
    return K.maximum(K.mean(K.sum(K.log(((A+gamma)/A)**2+((1/A)**2)*y_pred*y_pred),axis=-1,keepdims=True))-np.log(4),0)

def my_reciprocal_power(y_true, y_pred):
    return K.maximum(K.mean(K.pow(y_pred,-1)-1)-1,0)


def get_lr_metric(optimizer):
    # to print out the learning rate
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

def Rayleigh_Channel(x):
    # Rayleigh in tensor
    ch_coeff = K.sqrt(K.square(K.random_normal((1,),0,0.5))+K.square(K.random_normal((1,),0,0.5)))
    y = ch_coeff*x
    theta = K.random_uniform((1,),-np.pi/2,np.pi/2)
    real_0 = K.cos(theta) * y[:,0] - K.sin(theta)* y[:,1]
    imag_0 = K.cos(theta) * y[:,1] + K.sin(theta)* y[:,0]
    z = K.stack([real_0, imag_0], axis=1)

    return z
    
def Rayleigh(x, p):
    # Rayleigh in numpy
    ch_coeff = np.sqrt(np.add(np.square(np.random.normal(0,0.5, p)),np.square(np.random.normal(0,0.5, p))))
    y = np.multiply(np.tile(ch_coeff,2).reshape((p,2)),x)
    theta = np.random.uniform(-np.pi/2,np.pi/2, p)
    real_0 = np.cos(theta) * y[:,0] - np.sin(theta)* y[:,1]
    imag_0 = np.cos(theta) * y[:,1] + np.sin(theta)* y[:,0]
    z = np.stack([real_0, imag_0], axis=1)

    return z

def shuffleColumns(x):
    # Joint and marginal architectures
    return tf.gather(x, tf.random.shuffle(tf.range(tf.shape(x)[0])))

def data_generation_mi(data_x, data_y):
    # Create paired and unpaired samples
    data_xy = np.hstack((data_x, data_y))
    data_y_shuffle = np.take(data_y, np.random.permutation(data_y.shape[0]), axis=0)
    data_x_y = np.hstack((data_x, data_y_shuffle))
    return data_xy, data_x_y
    
class CORTICAL():
    def __init__(self, latent_dim, data_dim, alpha, channel, EbN0, power_constraint):
        # Input shape of z
        self.latent_dim = latent_dim

        # Shape of channel input x
        self.data_dim = data_dim

        # Joint architecture shape
        self.joint_dim = 2 * self.data_dim
        self.alpha = alpha
        self.channel = channel

        # Regularization coefficients
        self.power_constraint = power_constraint
        self.reg_PP = 1 if self.power_constraint=='PP' or self.power_constraint=='PPAP' else 0
        self.reg_AP = 1 if self.power_constraint=='AP' or self.power_constraint=='PPAP' else 0

        self.eps = np.sqrt(pow(10, -0.1 * EbN0) / (2 * 0.5))
        # Noise power
        N = self.eps**2

        optimizer_G = Adam(0.0002, 0.5)
        lr_metric = get_lr_metric(optimizer_G)
        optimizer_D = Adam(0.002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generators
        self.generator = self.build_generator()

        # The transmitter encodes the bits in s
        s_in = layers.Input(shape=(self.latent_dim,))
        x_n = self.generator(s_in)

        self.encoder = tf.keras.Model(s_in, x_n)


        if channel == 'AWGN' or 'MIMO':

            ch = layers.Lambda(lambda x: x)(x_n)
            y = layers.GaussianNoise(np.sqrt(N))(ch)  # AWGN layer

        elif channel == 'UNIF':

            ch = layers.Lambda(lambda x: x)(x_n)
            delta = np.sqrt(12*N) # variance of the uniform
            y = UniformNoise(-delta/2,delta/2)(ch)  # Uniform noise layer

        elif channel == 'AICN':

            ch = layers.Lambda(lambda x: x)(x_n)
            y = CauchyNoise(-np.pi/2,np.pi/2)(ch)  # Cauchy noise layer

        elif channel == 'RAY':
            print('Building the Rayleigh tensor channel')
            print('Regularization terms for peak and average power are %d and %d' %(self.reg_PP, self.reg_AP))

            ch = layers.Lambda(lambda x: Rayleigh_Channel(x))(x_n) # Fading
            y = layers.GaussianNoise(np.sqrt(N))(ch)  # AWGN layer

        elif channel == 'EXP':
            print('Building the exponential channel for the rayleigh amplitude')
            print('Regularization terms for peak and average power are %d and %d' %(self.reg_PP, self.reg_AP))

            #ch = layers.Lambda(lambda x: x**2+1)(x_n) # Amplification to x^2+1, old method
            ch = layers.Lambda(lambda x: K.pow(x,-1))(x_n) # use sigmoid output, x_n is s
            y = ExponentialNoise(0,1)(ch)  # Exponential layer
        else:
            print('Noise type is not defined, using AWGN')
            ch = layers.Lambda(lambda x: x)(x_n)
            y = layers.GaussianNoise(np.sqrt(N))(ch)  # AWGN layer

        xy = layers.Concatenate(name='network/concatenate_layer_1')([x_n, y])
        y_bar_input = layers.Lambda(lambda x: shuffleColumns(x))(y)  # shuffle y input as y_bar
        x_y = layers.Concatenate(name='network/concatenate_layer_2')([x_n, y_bar_input])

        # The discriminator takes as input joint or marginal vectors
        d_xy = self.discriminator(xy)
        d_x_y = self.discriminator(x_y)

        xy_in = layers.Input(shape=(self.joint_dim,))
        x_y_in = layers.Input(shape=(self.joint_dim,))

        d_xy_in = self.discriminator(xy_in)
        d_x_y_in = self.discriminator(x_y_in)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to help the discriminator

        self.combined_d = tf.keras.Model([xy_in,x_y_in], [d_xy_in,d_x_y_in])
        self.combined_d.compile(loss=[my_binary_crossentropy,wasserstein_loss],loss_weights=[self.alpha,1], optimizer=optimizer_D)

        # Update only the generator
        self.discriminator.trainable = False

        self.combined_g = tf.keras.Model(s_in, [d_xy, d_x_y, x_n, x_n])

        # for the Cauchy channel consider a different power constraint
        if self.channel == 'AICN':
            print('Creating generator with AICN constraints..')
            self.combined_g.compile(loss=[my_binary_crossentropy, wasserstein_loss, my_log_power, my_average_power], loss_weights=[self.alpha, 1, 1, 0],
                                optimizer=optimizer_G, metrics=[lr_metric])
        elif self.channel == 'EXP':
            print('Creating generator with EXP constraints..')
            self.combined_g.compile(loss=[my_binary_crossentropy, wasserstein_loss, my_reciprocal_power, my_average_power], loss_weights=[self.alpha, 1, 0.03, 0],
                                optimizer=optimizer_G, metrics=[lr_metric])
        elif self.channel == 'MIMO':
            print('Creating generator with MIMO elliptical constraints..')
            self.combined_g.compile(loss=[my_binary_crossentropy, wasserstein_loss, my_peak_power_ellipse, my_average_power], loss_weights=[self.alpha, 1, self.reg_PP, self.reg_AP],
                                optimizer=optimizer_G, metrics=[lr_metric])
        else:
            self.combined_g.compile(loss=[my_binary_crossentropy, wasserstein_loss, my_peak_power, my_average_power], loss_weights=[self.alpha, 1, self.reg_PP, self.reg_AP],
                                optimizer=optimizer_G, metrics=[lr_metric])


    def build_generator(self):

        model = tf.keras.models.Sequential()
        model.add(layers.Dense(512, activation="relu", input_dim=self.latent_dim))
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(512, activation="relu"))

        if self.channel == 'EXP':
            # due to constraints, see Faycal2001
            model.add(layers.Dense(self.data_dim, activation="sigmoid"))
        else:
            model.add(layers.Dense(self.data_dim))


        model.summary()

        T = layers.Input(shape=(self.latent_dim,))
        D = model(T)

        return tf.keras.Model(T, D)

    def build_discriminator(self):

        model = tf.keras.models.Sequential()
        model.add(layers.Dense(100, input_dim=self.joint_dim))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(100))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(1, activation='softplus'))

        model.summary()

        T = layers.Input(shape=(self.joint_dim,))
        D = model(T)

        return tf.keras.Model(T, D)

    def train(self, epochs, batch_size=40):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        epochs_D = 10
        epochs_G = 1
        MI_VAR = np.zeros((epochs,1))

        # set piecewise constant decay learning rate for the generator
        lr_g_values = [0.0002, 0.0002, 0.0002] # did not change it, for now
        boundary_epochs = [10000,50000] 

        for epoch in range(epochs):

            # Sample noise and train CORTICAL

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for epoch_D in range(epochs_D):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                ch_input = self.encoder.predict(noise)
                eps = self.eps

                if self.channel == 'AWGN' or 'MIMO':
                    ch_output = ch_input + eps * np.random.normal(0, 1, (batch_size, self.data_dim))
                elif self.channel == 'UNIF':
                    delta = np.sqrt(12)*eps
                    ch_output = ch_input + np.random.uniform(-delta/2, delta/2, (batch_size, self.data_dim))
                elif self.channel == 'AICN':
                    # P = k - gamma, C = log(1+P/gamma) or log(A/gamma) where gamma<k<= A
                    gamma = 1
                    ch_output = ch_input + gamma*np.tan(0.99*np.random.uniform(-np.pi/2, np.pi/2, (batch_size, self.data_dim)))
                elif self.channel == 'RAY':
                    ch_input_att = Rayleigh(ch_input, batch_size)
                    ch_output = ch_input_att + eps * np.random.normal(0, 1, (batch_size, self.data_dim))
                elif self.channel == 'EXP':
                    # ch_input_amp = np.power(ch_input, 2)+1 old method
                    ch_input_amp = np.power(ch_input, -1)
                    ch_output = -(ch_input_amp * np.log(1-np.random.uniform(0, 1, (batch_size, self.data_dim))))
                else:
                    # use AWGN
                    ch_output = ch_input + eps * np.random.normal(0, 1, (batch_size, self.data_dim))

                # Shuffle to implement pi(), see CORTICAL paper
                data_xy, data_x_y = data_generation_mi(ch_input, ch_output) # create paired and unpaired

                d_loss = self.combined_d.train_on_batch([data_xy, data_x_y],[valid,valid])

            # Print these values if desired
            D_value_xy = self.discriminator.predict(data_xy)
            D_value_x_y = self.discriminator.predict(data_x_y)
            J_e = self.alpha * np.log(D_value_xy) - D_value_x_y
            # Extract an estimate of the variational lower bound on the MI
            MI_VAR[epoch] = np.mean(J_e / self.alpha + 1 - np.log(self.alpha))

            # ---------------------
            #  Train Generator
            # ---------------------

            # check learning rates
            if any(epoch == c for c in boundary_epochs):
                print(boundary_epochs.index(epoch))
                K.set_value(self.combined_g.optimizer.learning_rate, lr_g_values[boundary_epochs.index(epoch)+1])

            for epoch_G in range(epochs_G):
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.combined_g.train_on_batch(noise, [valid, valid, valid, valid])

            # Plot the progress
            print("Current MI estimate is %f" % (MI_VAR[epoch]))
            print("CORTICAL training loss, epoch: %d [D1 loss: %f] [D2 loss: %f] [G1 loss: %f] [G2 loss: %f] [G3 loss: %f]" % (epoch, d_loss[1],d_loss[2], g_loss[1], g_loss[2], g_loss[3]))
            if epoch%1000 == 0:
                noise_test = np.random.normal(0, 1, (10000, self.latent_dim))
                partial_ch_input = self.encoder.predict(noise_test)
                sio.savemat('data_CORTICAL_%d.mat'%epoch,{'ch_input': partial_ch_input})
            if epoch==epochs-1:
                # save the evolution of the MI over training
                sio.savemat('data_CORTICAL_%d.mat'%epoch,{'ch_input': partial_ch_input, 'MI_VAR_training': MI_VAR})

    def test(self, test_size=1000):

        # test now..
        noise = np.random.normal(0,1, (test_size, self.latent_dim))

        # Get channel input data
        ch_input = self.encoder.predict(noise)

        eps = self.eps

        if self.channel == 'AWGN' or 'MIMO':
            ch_output = ch_input + eps * np.random.normal(0, 1, (test_size, self.data_dim))
        elif self.channel == 'UNIF':
            delta = np.sqrt(12)*eps
            ch_output = ch_input + np.random.uniform(-delta/2, delta/2, (test_size, self.data_dim))
        elif self.channel == 'AICN':
            gamma = 1
            ch_output = ch_input + gamma*np.tan(0.99*np.random.uniform(-np.pi/2, np.pi/2, (test_size, self.data_dim)))
        elif self.channel == 'RAY':
            ch_input_att = Rayleigh(ch_input, test_size)
            ch_output = ch_input_att + eps * np.random.normal(0, 1, (test_size, self.data_dim))
        elif self.channel == 'EXP':
            # ch_input_amp = np.power(ch_input, 2)+1
            ch_input_amp = np.power(ch_input, -1)
            ch_output = -(ch_input_amp * np.log(1-np.random.uniform(0, 1, (test_size, self.data_dim))))
        else:
            ch_output = ch_input + eps * np.random.normal(0, 1, (test_size, self.data_dim))

        
        # Shuffle to implement pi(), see CORTICAL paper
        data_xy, data_x_y = data_generation_mi(ch_input, ch_output)

        # for the variational lower bound, you need to average over the joint samples and the marginals
        D_value_1 = self.discriminator.predict(data_xy)
        D_value_2 = self.discriminator.predict(data_x_y)
        J_e = self.alpha * np.log(D_value_1) - D_value_2
        MI_VAR = J_e / self.alpha + 1 - np.log(self.alpha)


        # Return also the constellation
        return MI_VAR, ch_input, ch_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', help='Number of channel samples to train on at once', default=32)
    parser.add_argument('--epochs', help='Number of epochs to train for', default=100001)
    parser.add_argument('--test_size', help='Number of data samples for testing', default=10000)
    parser.add_argument('--dim', help='Channel input dimension', default=1)
    parser.add_argument('--alpha', help='Alpha parameter of CORTICAL', default=1)
    parser.add_argument('--channel', help='Type of channel: AWGN, UNIF, AICN, EXP, RAY, MIMO', default='AWGN')
    parser.add_argument('--power_constraint', help='Type of constraint: PP (peak power), AP (amplitude power), PPAP', default='PP')


    args = parser.parse_args()

    test_size = int(args.test_size)
    data_dim = int(args.dim)
    alpha = float(args.alpha)
    channel = str(args.channel)
    power_constraint = str(args.power_constraint)

    EbN0_dB = range(-14, 29) # SNR range

    latent_dim = 30 # z input dimension

    MI_VAR_total = np.zeros((len(EbN0_dB),test_size))

    features_x = []
    features_y = []

    j = 0
    for EbN0 in EbN0_dB:
        print("Current EbN0 is: %f" %EbN0)
        # Initialize CORTICAL
        cortical = CORTICAL(latent_dim, data_dim, alpha, channel, EbN0, power_constraint)
        # Train
        cortical.train(epochs=int(args.epochs), batch_size=int(args.batch_size))
        # Test
        MI_VAR, data_x, data_y = cortical.test(test_size)
        MI_VAR_total[j, :] = np.transpose(MI_VAR)
        features_x.append(data_x)
        features_y.append(data_y)
        del cortical
        j = j + 1

    sio.savemat('data_CORTICAL.mat', {'MI_VAR': MI_VAR_total, 'ch_input': features_x, 'ch_output':features_y})
