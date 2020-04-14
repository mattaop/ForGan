from models.feed_forward_gan import GAN, WGAN
from models.conv_gan import ConvGAN, ConvWGAN
from models.recurrent_gan import RecurrentGAN, RecurrentWGAN, RecurrentConvGAN
from models.reccurent_neural_network import RNN


def get_GAN(model_name: str = 'gan')-> {GAN.GAN, WGAN.WGAN, ConvGAN.ConvGAN, ConvWGAN.ConvWGAN,
                                        RecurrentGAN.RecurrentGAN, RecurrentWGAN.RecurrentWGAN,
                                        RecurrentConvGAN.RecurrentConvGAN}:
    if model_name.lower() == 'gan':
        print('Model: GAN')
        model = GAN.GAN()
    elif model_name.lower() == 'wgan':
        print('Model: WGAN')
        model = WGAN.WGAN()
    elif model_name.lower() == 'convgan':
        print('Model: ConvGAN')
        model = ConvGAN.ConvGAN()
    elif model_name.lower() == 'convwgan':
        print('Model: ConvWGAN')
        model = ConvWGAN.ConvWGAN()
    elif model_name.lower() == 'recurrentgan':
        print('Model: RecurrentGAN')
        model = RecurrentGAN.RecurrentGAN()
    elif model_name.lower() == 'recurrentwgan':
        print('Model: RecurrentWGAN')
        model = RecurrentWGAN.RecurrentWGAN()
    elif model_name.lower() == 'recurrentconvgan':
        print('Model: RecurrentConvGAN')
        model = RecurrentConvGAN.RecurrentConvGAN()
    else:
        ImportError('Model ' + model_name + 'not found')
        model = None
    return model


def get_model(model_name: str = 'rnn'):
    return RNN.RNN()
