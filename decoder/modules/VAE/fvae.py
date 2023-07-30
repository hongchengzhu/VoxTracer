import numpy as np
import torch
import torch.distributions as dist
from torch import nn

from modules.VAE.conv import ConditionalConvBlocks
from modules.VAE.res_flow import ResFlow
from modules.wavenet import WN


# class FVAEEncoder(nn.Module):
#     def __init__(self, c_in, hidden_size, c_latent, kernel_size,
#                  n_layers, c_cond=0, p_dropout=0, strides=[4], nn_type='wn'):
#         super().__init__()
#         self.strides = strides
#         self.hidden_size = hidden_size
#         # if np.prod(strides) == 1:
#         #     self.pre_net = nn.Conv1d(4, 128, kernel_size=1)
#         # else:
#         #     self.pre_net = nn.Sequential(*[
#         #         nn.Conv1d(4, 128, kernel_size=s * 2, stride=s, padding=s // 2)
#         #         if i == 0 else
#         #         nn.Conv1d(4, 128, kernel_size=s * 2, stride=s, padding=s // 2)
#         #         for i, s in enumerate(strides)
#         #     ])
#             # self.pre_net = nn.Conv1d(c_in, hidden_size, kernel_size=1)
#         self.pre_net = nn.Conv1d(4, 128, kernel_size=1)
#         if nn_type == 'wn':
#             self.nn = WN(hidden_size, kernel_size, 1, n_layers, c_cond, p_dropout)
#         elif nn_type == 'conv':
#             self.nn = ConditionalConvBlocks(
#                 hidden_size, c_cond, hidden_size, None, kernel_size,
#                 layers_in_block=2, is_BTC=False, num_layers=n_layers)
#
#         self.out_proj = nn.Conv1d(128, 8, 1)
#         self.latent_channels = 4
#
#     def forward(self, x):
#         x = self.pre_net(x)     # unsample from [batch, 4, 2048] to [batch, 128, 2048]
#         # nonpadding = nonpadding[:, :, ::np.prod(self.strides)][:, :, :x.shape[-1]]
#         # nonpadding = torch.ones(x.shape)
#         # x = x * nonpadding
#         x = self.nn(x)
#         x = self.out_proj(x)    # downsample from [batch, 128, 2048] to [batch, 4, 2048]
#         m, logs = torch.split(x, self.latent_channels, dim=1)
#         z = (m + torch.randn_like(m) * torch.exp(logs))
#         return z, m, logs


class FVAEDecoder(nn.Module):
    def __init__(self, c_latent, hidden_size, out_channels, kernel_size,
                 n_layers, c_cond=0, p_dropout=0, strides=[4], nn_type='wn'):
        super().__init__()
        # self.strides = strides
        # self.hidden_size = hidden_size
        # self.pre_net = nn.Sequential(*[
        #     nn.ConvTranspose1d(c_latent, hidden_size, kernel_size=s, stride=s)
        #     if i == 0 else
        #     nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=s, stride=s)
        #     for i, s in enumerate(strides)
        # ])
        self.pre_net = nn.Conv1d(4, 128, kernel_size=1)
        # self.pre_net = nn.ConvTranspose1d(c_latent, hidden_size, kernel_size=1, stride=1)
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


class FVAE(nn.Module):
    def __init__(self,
                 c_in_out, hidden_size, c_latent,
                 kernel_size, enc_n_layers, dec_n_layers, c_cond, strides,
                 encoder_type='wn', decoder_type='wn'):
        super(FVAE, self).__init__()

        self.decoder = FVAEDecoder(c_latent, hidden_size, c_in_out, kernel_size,
                                   dec_n_layers, c_cond, strides=strides, nn_type=decoder_type)
        # self.prior_dist = dist.Normal(0, 1)

    def forward(self, x=None, cond=None, infer=False, noise_scale=1.0):
        """

        :param x: [B, C_in_out, T]
        :param nonpadding: [B, 1, T]
        :param cond: [B, C_g, T]
        :return:
        """
        # if nonpadding is None:
        #     nonpadding = torch.ones(x.shape[0], 1, x.shape[2])
        # cond_sqz = self.g_pre_net(cond)
        # cond_sqz = cond
        # if not infer:
        #     z_q, m_q, logs_q = self.encoder(x)
        #     q_dist = dist.Normal(m_q, logs_q.exp())
        #     # if self.use_prior_flow:
        #     #     logqx = q_dist.log_prob(z_q)
        #     #     z_p = self.prior_flow(z_q, nonpadding_sqz, cond_sqz)
        #     #     logpx = self.prior_dist.log_prob(z_p)
        #     #     loss_kl = ((logqx - logpx) * nonpadding_sqz).sum() / nonpadding_sqz.sum() / logqx.shape[1]
        #     # else:
        #     loss_kl = torch.distributions.kl_divergence(q_dist, self.prior_dist)
        #     loss_kl = loss_kl.sum() / (z_q.shape[0] * z_q.shape[1] * z_q.shape[2])
        #     z_p = None
        #
        #     return z_q, loss_kl, z_p, m_q, logs_q
        # else:
        # latent_shape = [cond_sqz.shape[0], self.latent_size, cond_sqz.shape[2]]
        # z_p = torch.randn(latent_shape).to(cond.device) * noise_scale
        # # if self.use_prior_flow:
        # #     z_p = self.prior_flow(z_p, 1, cond_sqz, reverse=True)
        # return z_p
        return None
