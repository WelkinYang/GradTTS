from tqdm import tqdm
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F

import modules
import commons
import attentions
import monotonic_align
import unet

class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = attentions.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = attentions.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self, 
      n_vocab, 
      out_channels, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      n_heads, 
      n_layers, 
      kernel_size, 
      p_dropout, 
      window_size=None,
      block_length=None,
      mean_only=False,
      prenet=False,
      gin_channels=0):

    super().__init__()

    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.prenet = prenet
    self.gin_channels = gin_channels

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    if prenet:
      self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      window_size=window_size,
      block_length=block_length,
    )

    self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
    if not mean_only:
      self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)
  
  def forward(self, x, x_lengths, g=None):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if self.prenet:
      x = self.pre(x, x_mask)
    x = self.encoder(x, x_mask)

    if g is not None:
      g_exp = g.expand(-1, -1, x.size(-1))
      x_dp = torch.cat([torch.detach(x), g_exp], 1)
    else:
      x_dp = torch.detach(x)

    x_m = self.proj_m(x) * x_mask
    if not self.mean_only:
      x_logs = self.proj_s(x) * x_mask
    else:
      x_logs = torch.zeros_like(x_m)

    logw = self.proj_w(x_dp, x_mask)
    return x_m, x_logs, logw, x_mask


class DiffusionDecoder(nn.Module):
  def __init__(self, 
      unet_channels=64,
      unet_in_channels=2,
      unet_out_channels=1,
      dim_mults=(1, 2, 4),
      groups=8,
      with_time_emb=True,
      beta_0=0.05,
      beta_1=20,
      N=1000,
      T=1):

    super().__init__()

    self.beta_0 = beta_0
    self.beta_1 = beta_1
    self.N = N
    self.T = T
    self.delta_t = T*1.0 / N
    self.discrete_betas = torch.linspace(beta_0, beta_1, N)
    self.unet = unet.Unet(dim=unet_channels, out_dim=unet_out_channels, dim_mults=dim_mults, groups=groups, channels=unet_in_channels, with_time_emb=with_time_emb)

  def marginal_prob(self, mu, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None]) * x + (1-torch.exp(log_mean_coeff[:, None, None]) ) * mu
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def cal_loss(self, x, mu, t, z, std, g=None):
    time_steps = t * (self.N - 1)
    if g:
        x = torch.stack([x, mu, g], 1)
    else:
        x = torch.stack([x, mu], 1)
    grad = self.unet(x, time_steps)
    loss = torch.square(grad + z / std[:, None, None]) * torch.square(std[:, None, None])
    return loss

  def forward(self, mu, y=None, g=None, gen=False):
    if not gen:
      t = torch.FloatTensor(y.shape[0]).uniform_(0, self.T).to(y.device)  # sample a random t
      mean, std = self.marginal_prob(mu, y, t)
      z = torch.randn_like(y)
      x = mean + std[:, None, None] * z
      loss = self.cal_loss(x, mu, t, z, std, g)
      return loss
    else:
      with torch.no_grad():
        y_T = torch.randn_like(mu) + mu
        y_t_plus_one = y_T
        y_t = None
        for n in tqdm(range(self.N - 1, -1, -1)):
          t = torch.FloatTensor(1).fill_(n).to(mu.device)
          if g:
              x = torch.stack([y_t_plus_one, mu, g], 1)
          else:
              x = torch.stack([y_t_plus_one, mu], 1)
          grad = self.unet(x, t)
          y_t = y_t_plus_one-0.5*self.delta_t*self.discrete_betas[n]*(mu-y_t_plus_one-grad)
          y_t_plus_one = y_t

      return y_t


class DiffusionGenerator(nn.Module):
  def __init__(self, 
      n_vocab, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      enc_out_channels,
      kernel_size=3, 
      n_heads=2, 
      n_layers_enc=6,
      p_dropout=0., 
      n_speakers=0, 
      gin_channels=0, 
      window_size=None,
      block_length=None,
      mean_only=False,
      hidden_channels_enc=None,
      hidden_channels_dec=None,
      prenet=False,
      dec_unet_channels=64,
      dec_dim_mults=(1, 2, 4),
      dec_groups=8,
      dec_unet_in_channels=2,
      dec_unet_out_channels=1,
      dec_with_time_emb=True,
      beta_0=0.05,
      beta_1=20,
      N=1000,
      T=1,
      **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.enc_out_channels = enc_out_channels
    self.dec_in_channels = enc_out_channels
    self.kernel_size = kernel_size
    self.n_heads = n_heads
    self.n_layers_enc = n_layers_enc
    self.p_dropout = p_dropout
    self.n_speakers = n_speakers
    self.gin_channels = enc_out_channels
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.hidden_channels_enc = hidden_channels_enc
    self.hidden_channels_dec = hidden_channels_dec
    self.prenet = prenet
    self.dec_unet_channels = dec_unet_channels
    self.dec_unet_in_channels = dec_unet_in_channels if self.n_speakers < 1 else dec_unet_in_channels+1
    self.dec_unet_out_channels = dec_unet_out_channels
    self.dec_dim_mults = dec_dim_mults
    self.dec_groups = dec_groups
    self.dec_with_time_emb = dec_with_time_emb
    self.beta_0 = beta_0
    self.beta_1 = beta_1
    self.N = N
    self.T = T

    self.encoder = TextEncoder(
        n_vocab, 
        enc_out_channels, 
        hidden_channels_enc or hidden_channels, 
        filter_channels, 
        filter_channels_dp, 
        n_heads, 
        n_layers_enc, 
        kernel_size, 
        p_dropout, 
        window_size=window_size,
        block_length=block_length,
        mean_only=mean_only,
        prenet=prenet,
        gin_channels=gin_channels)

    self.decoder = DiffusionDecoder(
        unet_channels=self.dec_unet_channels,
        unet_in_channels=self.dec_unet_in_channels,
        unet_out_channels=self.dec_unet_out_channels,
        dim_mults=self.dec_dim_mults,
        groups=self.dec_groups,
        with_time_emb=self.dec_with_time_emb,
        beta_0=self.beta_0,
        beta_1=self.beta_1,
        N=self.N,
        T=self.T)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
      nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]
    x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)

    if gen:
      w = torch.exp(logw) * x_mask * length_scale
      w_ceil = torch.ceil(w)
      y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
      y_max_length = None
    else:
      y_max_length = y.size(2)
    #y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    if gen:
      attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
      z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2)
      logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

      y = self.decoder(z_m, gen=True)
      return (y, z_m, z_logs, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
    else:
      with torch.no_grad():
        x_s_sq_r = torch.exp(-2 * x_logs)
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
        logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (y ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), y) # [b, t, d] x [b, d, t'] = [b, t, t']
        logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
        logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

        attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
      z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
      grad_loss = self.decoder(mu=z_m, y=y, g=g, gen=False).mean()
      return grad_loss, (z_m, z_logs, z_mask), (attn, logw, logw_)
