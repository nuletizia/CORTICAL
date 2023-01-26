from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Lambda
import tensorflow as tf
import math

def tan(x):
    return tf.math.tan(x)

class CauchyNoise(Layer):
    """Apply additive Cauchy noise
    Only active at training time since it is a regularization layer.

    # Arguments
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution

    # Input shape
        Arbitrary.

    # Output shape
        Same as the input shape.
    """

    def __init__(self, minval=-0.5, maxval=0.5, gamma = 1, **kwargs):
        super(CauchyNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.minval = minval
        self.maxval = maxval
        self.gamma = gamma

    def call(self, inputs, training=None):
        def noised():
            sc_noise = K.random_uniform(shape=K.shape(inputs),minval=self.minval, maxval=self.maxval)
            cauchy_noise = Lambda(lambda x: tan(x))(0.98*sc_noise) #to avoid errors
            return inputs + self.gamma*cauchy_noise

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'minval': self.minval, 'maxval': self.maxval}
        base_config = super(PolynomialNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
