import math
import torch
import torch.nn as nn

from diffsynth.util import resample_frames
from diffsynth.layers import Resnet1D, Normalize2d
from diffsynth.transforms import LogTransform
from nnAudio.Spectrogram import MelSpectrogram, MFCC

from diffsynth.f0 import FMIN, FMAX

class MelEstimator(nn.Module):
    def __init__(self, output_dim, n_mels=128, n_fft=1024, hop=256, sample_rate=16000, channels=64, kernel_size=7, strides=[2,2,2], num_layers=1, hidden_size=512, dropout_p=0.0, bidirectional=False, norm='batch'):
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.logmel = nn.Sequential(MelSpectrogram(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop, center=True, power=1.0, htk=True, trainable_mel=False, trainable_STFT=False), LogTransform())
        self.norm = Normalize2d(norm) if norm else None
        # Regular Conv
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(1, channels, kernel_size,
                        padding=kernel_size // 2,
                        stride=strides[0]), nn.BatchNorm1d(channels), nn.ReLU())]
            + [nn.Sequential(nn.Conv1d(channels, channels, kernel_size,
                         padding=kernel_size // 2,
                         stride=strides[i]), nn.BatchNorm1d(channels), nn.ReLU())
                         for i in range(1, len(strides))])
        self.l_out = self.get_downsampled_length()[-1] # downsampled in frequency dimension
        print('output dims after convolution', self.l_out)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(self.l_out * channels, hidden_size, num_layers=num_layers, dropout=dropout_p, batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_dim)
        self.output_dim = output_dim

    def forward(self, audio):
        x = self.logmel(audio)
        x = self.norm(x)
        batch_size, n_mels, n_frames = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.n_mels).unsqueeze(1)
        # x: [batch_size*n_frames, 1, n_mels]
        for i, conv in enumerate(self.convs):
            x = conv(x)
        x = x.view(batch_size, n_frames, self.channels, self.l_out)
        x = x.view(batch_size, n_frames, -1)
        D = 2 if self.bidirectional else 1
        output, _hidden = self.gru(x, torch.zeros(D * self.num_layers, batch_size, self.hidden_size, device=x.device))
        # output: [batch_size, n_frames, self.output_dim]
        output = self.out(output)
        return torch.sigmoid(output)

    def get_downsampled_length(self):
        l = self.n_mels
        lengths = [l]
        for conv in self.convs:
            conv_module = conv[0]
            l = (l + 2 * conv_module.padding[0] - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1) - 1) // conv_module.stride[0] + 1
            lengths.append(l)
        return lengths