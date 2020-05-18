from models.feed_forward_gan import GAN, WGAN, DSGAN
from models.conv_gan import ConvGAN, ConvWGAN
from models.recurrent_gan import RecurrentGAN, RecurrentWGAN, RecurrentDSGAN, RecurrentConvGAN, RecurrentConvWGAN, \
    RecurrentConvDSGAN
from models.baseline_models import RNN
from models.hybrid_gan import ESWGAN, ESRecurrentWGAN


def get_gan(cfg):
    model_name = cfg['model_name']
    loss = cfg['loss_function']
    model = None
    if model_name.lower() == 'gan':
        if loss.lower() in ['kl', 'kl-div', 'kl-divergence', 'kl divergence', None]:
            print('Model: GAN, Loss: Binary Crossentropy loss')
            model = GAN.GAN(cfg)
        elif loss.lower() in ['wasserstein', 'w']:
            print('Model: GAN, Loss: Wasserstein loss')
            model = WGAN.WGAN(cfg)
        elif loss.lower() in ['diversity-sensitive', 'diversity sensitive', 'ds']:
            print('Model: GAN, Loss: Diversity-Sensitive loss')
            model = DSGAN.DSGAN(cfg)
        else:
            print('Model GAN has no loss:', loss)
            AttributeError()
    elif model_name.lower() == 'convgan':
        if loss.lower() in ['kl', 'kl-div', 'kl-divergence', 'kl divergence', None]:
            print('Model: ConvGAN, Loss: Binary Crossentropy')
            model = ConvGAN.ConvGAN(cfg)
        elif loss.lower() in ['wasserstein', 'w']:
            print('Model: ConvGAN, Loss: Wasserstein loss')
            model = ConvWGAN.ConvWGAN(cfg)
        else:
            print('Model GAN has no loss:', loss)
            AttributeError()

    elif model_name.lower() == 'recurrentgan':
        if loss.lower() in ['kl', 'kl-div', 'kl-divergence', 'kl divergence', None]:
            print('Model: RecurrentGAN, Loss: Binary Crossentropy loss')
            model = RecurrentGAN.RecurrentGAN(cfg)
        elif loss.lower() in ['wasserstein', 'w']:
            print('Model: RecurrentGAN, Loss: Wasserstein loss')
            model = RecurrentWGAN.RecurrentWGAN(cfg)
        elif loss.lower() in ['diversity-sensitive', 'diversity sensitive', 'ds']:
            print('Model: RecurrentGAN, Loss: Diversity-Sensitive loss')
            model = RecurrentDSGAN.RecurrentDSGAN(cfg)
        else:
            print('Model GAN has no loss:', loss)
            AttributeError()
    elif model_name.lower() == 'recurrentconvgan':
        if loss.lower() in ['kl', 'kl-div', 'kl-divergence', 'kl divergence', None]:
            print('Model: RecurrentConvGAN, Loss: Binary Crossentropy loss')
            model = RecurrentConvGAN.RecurrentConvGAN(cfg)
        elif loss.lower() in ['wasserstein', 'w']:
            print('Model: RecurrentGAN, Loss: Wasserstein loss')
            model = RecurrentConvWGAN.RecurrentConvWGAN(cfg)
        elif loss.lower() in ['diversity-sensitive', 'diversity sensitive', 'ds']:
            print('Model: RecurrentGAN, Loss: Diversity-Sensitive loss')
            model = RecurrentConvDSGAN.RecurrentConvDSGAN(cfg)
        else:
            print('Model GAN has no loss:', loss)
            AttributeError()
    elif model_name.lower() == 'esgan':
        if loss.lower() in ['kl', 'kl-div', 'kl-divergence', 'kl divergence', None]:
            print('Model: ES-WGAN, Loss: Binary Crossentropy loss')
            model = ESWGAN.ESWGAN(cfg)
        elif loss.lower() in ['wasserstein', 'w']:
            print('Model: ES-RecurrentWGAN, Loss: Wasserstein loss')
            model = ESRecurrentWGAN.ESRecurrentWGAN(cfg)

        else:
            print('Model GAN has no loss:', loss)
            AttributeError()
    elif model_name.lower() == 'rnn':
        print('Model: RNN')
        model = RNN.RNN(cfg)
    else:
        ImportError('Model ' + model_name + 'not found')
    return model
