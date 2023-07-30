import torch.nn as nn
from modules.VAE.fvae import FVAE
from torch import nn
from modules.VAE.conv import ConditionalConvBlocks
from modules.wavenet import WN


class FVAEDecoder(nn.Module):
    def __init__(self, hidden_size=128, kernel_size=5,
                 n_layers=4, c_cond=192, p_dropout=0, nn_type='wn'):
        super().__init__()
        self.pre_net = nn.Conv1d(4, 128, kernel_size=1)
        if nn_type == 'wn':
            self.nn = WN(hidden_size, kernel_size, 1, n_layers, c_cond, p_dropout)
        elif nn_type == 'conv':
            self.nn = ConditionalConvBlocks(
                hidden_size, c_cond, hidden_size, [1] * n_layers, kernel_size,
                layers_in_block=2, is_BTC=False)
        self.out_proj = nn.Conv1d(hidden_size, 4, 1)

    def forward(self, x):
        x = self.pre_net(x)
        # x = x * nonpadding
        x = self.nn(x)
        x = self.out_proj(x)
        return x



class Generator(nn.Module):
    """Generator network."""
    def __init__(self):
        super(Generator, self).__init__()

        self.fvae = FVAE(c_in_out=128, hidden_size=128, c_latent=16,
                         kernel_size=5, enc_n_layers=8,
                         dec_n_layers=4, c_cond=192, strides=[4],
                         encoder_type='wn',
                         decoder_type='wn')

    def forward(self, z):

        emb_hat = self.fvae.decoder(z)
        return emb_hat


