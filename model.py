import torch
from torch import nn


class Denoiser(torch.nn.Module):
  # D: no of encoder layers
  # H: no. of output channels in first layer
  # K: kernel size
  # S: stride
  def __init__(
  self, 
  n_layers, 
  output_channels, 
  chin=1,
  chout=1,
  hidden=48,
  depth=5,
  N_attention = 3,
  kernel_size=8,
  stride=4,
  causal=True,
  resample=4,
  growth=2,
  max_hidden=10_000,
  normalize=True,
  glu=True,
  rescale=0.1,
  floor=1e-3,
  sample_rate=22_050
  ):
    super(Denoiser, self).__init__()
    self.D = n_layers
    self.H = output_channels
    self.K = kernel_size
    self.S = kernel_size // 2

    self.encoder = nn.ModuleList([])
    self.decoder = nn.ModuleList([])
    self.attention = nn.ModuleList([])
    activation = nn.GLU(1) if glu else nn.ReLU()
    ch_scale = 2 if glu else 1

    for index in range(depth):

      encode = []
      encode += [
          nn.Conv1d(chin, hidden, kernel_size, stride),
          nn.ReLU(),
          nn.Conv1d(hidden, hidden * ch_scale, 1),
          activation,
      ]
      self.encoder.append(nn.Sequential(*encode))

      decode = []
      decode += [
          nn.Conv1d(hidden, ch_scale * hidden, 1),
          activation,
          nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
      ]
      if index > 0:
          decode.append(nn.ReLU())
      self.decoder.insert(0, nn.Sequential(*decode))
      chout = hidden
      chin = hidden
      hidden = min(int(growth * hidden), max_hidden)

    for i in range(N_attention):
      attention = []
      attention += [
        nn.MultiheadAttention(embed_dim=chin, num_heads=8),
        nn.Linear(chin, 2*chin),
        nn.Linear(2*chin, chin)
      ]
      self.attention.append(nn.Sequential(*attention))

  def forward(self, input):
    length = input.shape[-1]
    skip_outputs = []
    for encoder in self.encoder:
      x = encoder(x)
      skip_outputs.append(x)

    # x = x.permute(2, 0, 1)
    # x, _ = self.attention(x)
    # x = x.permute(1, 2, 0)
    for attention in self.attention:
      x, _ = attention(x, x, x)
    
    for decode in self.decoder:
        skip = skip_outputs.pop(-1)
        x = x + skip[..., :x.shape[-1]]
        x = decode(x)

    return x