from old_files.Old_GAN import GAN


class ForGAN(GAN):
    def __init__(self):
        GAN.__init__(self)


if __name__ == '__main__':
    gan = ForGAN()
    gan.train(epochs=1500, batch_size=128, discriminator_epochs=3)
    # gan.forecast(steps=100)
    gan.monte_carlo_forecast(steps=50, mc_forward_passes=1000)
