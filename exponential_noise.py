from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Lambda

class ExponentialNoise(Layer):
    """Apply exponential noise
    Only active at training time since it is a regularization layer.

    # Arguments
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution

    # Input shape
        Arbitrary.

    # Output shape
        Same as the input shape.
    """

    def __init__(self, minval=0, maxval=1, **kwargs):
        super(ExponentialNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.minval = minval
        self.maxval = maxval

    def call(self, inputs, training=None):
        def noised():
            exp_noise = -K.log(1-K.random_uniform(shape=K.shape(inputs),minval=self.minval, maxval=self.maxval))
            return inputs*exp_noise

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'minval': self.minval, 'maxval': self.maxval}
        base_config = super(ExponentialNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
