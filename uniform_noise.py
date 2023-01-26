from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K


class UniformNoise(Layer):
    """Apply additive uniform noise
    Only active at training time since it is a regularization layer.

    # Arguments
        minval: Minimum value of the uniform distribution
        maxval: Maximum value of the uniform distribution

    # Input shape
        Arbitrary.

    # Output shape
        Same as the input shape.
    """

    def __init__(self, minval=-1.0, maxval=1.0, **kwargs):
        super(UniformNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.minval = minval
        self.maxval = maxval

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_uniform(shape=K.shape(inputs),
                                             minval=self.minval,
                                             maxval=self.maxval)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'minval': self.minval, 'maxval': self.maxval}
        base_config = super(UniformNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
