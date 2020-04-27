from models.feed_forward_gan import GAN, WGAN
from models.conv_gan import ConvGAN, ConvWGAN
from models.recurrent_gan import RecurrentGAN, RecurrentWGAN, RecurrentConvGAN, RecurrentConvWGAN
from models.reccurent_neural_network import RNN


def get_gan(cfg):
    model_name = cfg['model_name']
    if model_name.lower() == 'gan' and not cfg['wasserstein_loss']:
        print('Model: GAN')
        model = GAN.GAN(cfg)
    elif model_name.lower() == 'gan' and cfg['wasserstein_loss']:
        print('Model: WGAN')
        model = WGAN.WGAN(cfg)
    elif model_name.lower() == 'convgan' and not cfg['wasserstein_loss']:
        print('Model: ConvGAN')
        model = ConvGAN.ConvGAN(cfg)
    elif model_name.lower() == 'convgan' and cfg['wasserstein_loss']:
        print('Model: ConvWGAN')
        model = ConvWGAN.ConvWGAN(cfg)
    elif model_name.lower() == 'recurrentgan' and not cfg['wasserstein_loss']:
        print('Model: RecurrentGAN')
        model = RecurrentGAN.RecurrentGAN(cfg)
    elif model_name.lower() == 'recurrentgan' and cfg['wasserstein_loss']:
        print('Model: RecurrentWGAN')
        model = RecurrentWGAN.RecurrentWGAN(cfg)
    elif model_name.lower() == 'recurrentconvgan' and not cfg['wasserstein_loss']:
        print('Model: RecurrentConvGAN')
        model = RecurrentConvGAN.RecurrentConvGAN(cfg)
    elif model_name.lower() == 'recurrentconvgan' and cfg['wasserstein_loss']:
        print('Model: RecurrentConvWGAN')
        model = RecurrentConvWGAN.RecurrentConvWGAN(cfg)
    elif model_name.lower() == 'rnn':
        model = RNN.RNN(cfg)
    else:
        ImportError('Model ' + model_name + 'not found')
        model = None
    return model


def get_model(model_name: str = 'rnn'):
    return RNN.RNN()
