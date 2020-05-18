from keras import Model
from keras.layers import *

from utility.ClipConstraint import ClipConstraint
from models.recurrent_gan.RecurrentDSGAN import RecurrentDSGAN


class RecurrentConvDSGAN(RecurrentDSGAN):
    def __init__(self, cfg):
        RecurrentDSGAN.__init__(self, cfg)
        self.plot_folder = 'RecurrentConvDSGAN'

    def build_discriminator(self):
        historic_shape = (self.window_size, 1)
        future_shape = (self.output_size, 1)

        historic_inp = Input(shape=historic_shape)
        future_inp = Input(shape=future_shape)

        x = Concatenate(axis=1)([historic_inp, future_inp])

        # define the constraint

        x = Conv1D(32, kernel_size=4)(x)
        x = LeakyReLU(alpha=0.1)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = MaxPooling1D(pool_size=2)(x)
        # x = Conv1D(64, kernel_size=4, kernel_constraint=const)(x)
        # x = LeakyReLU(alpha=0.1)(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        # x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(32)(x)
        x = LeakyReLU(alpha=0.1)(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[historic_inp, future_inp], outputs=validity)
        model.summary()

        return model
