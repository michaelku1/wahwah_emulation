import logging
import sys
import os
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import numpy as np

import pytorch_lightning as pl
import auraloss
# BUG
# from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset, random_permute

from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset, random_permute
from processors.parametric_eq import parametric_eq, load_eq_parameters
from processors.helmut_keller import wah_wah
from scipy import fftpack

import yaml
from typing import Dict, Optional, List, Any, Tuple, Type, Union
import pyloudnorm as pyln
import importlib

from tqdm import tqdm
from collections import defaultdict
import tensorboard

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

class ParameterNetwork(torch.nn.Module):
    """
    uses MLP layers to map knobs to controls
    """

    def __init__(self, num_knobs, num_control_params: int, **kwargs):
        super().__init__()
        self.num_control_params = num_control_params
        self.linear1 = torch.nn.Linear(num_knobs, 128) # depends on input knob
        self.linear2 = torch.nn.Linear(128, 256)
        self.linear3 = torch.nn.Linear(256, num_control_params) # depends on processor input params

    def forward(self, knobs):

        y = self.linear1(knobs)
        y = self.linear2(y)
        y = self.linear3(y)

        y = torch.sigmoid(y)

        return y


class Weiner_Hammerstein(torch.nn.Module):

    """
    W-H baseline model for reverse engineering, processing samples by batch
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, L1_params, L2_params, parametric_eq, fx_processor, p_hat, sr=44100):
        """
        Args:
            x: raw audio (1, bs, sample_length)
            L1_params: first linear filter parameters
            L2_params: second linear filter parameters
            parametric_eq: parametric_eq function
            fx_processor: processor class
            p_hat: NN output DSP controls


        returns:
            x: rendered audio (bs, ch, sample_length)

        """

        # out x: (bs, 1, sample_length)
        x = parametric_eq(x, sr, **L1_params)

        # out x: (bs, sample_length)
        x = fx_processor.process_normalized(x, p_hat)

        # print("x shape after fx processor", x.shape)

        # out x: (bs, 1, sample_length)
        x = x.view(x.shape[0], 1, x.shape[-1])

        # out x: (bs, 1, sample_length)
        x = parametric_eq(x, sr, **L2_params)

        # out x: (1, bs, sample_length)
        x = x.transpose(0,1)

        return x

test_input_paths_clean = "./dataset1/test/dry/"
test_input_paths_wet = "./dataset1/test/wet/"
base_path = "./dataset1/test/dry" # for obtaining file metadata info
config_path = "./configs/config1.yml"
preset_path = "./dataset1/preset_config1.yml"
l1_config_path = "./configs/parametric_eq1.yml"
l2_config_path = "./configs/parametric_eq2.yml"

# saved model path
model_path = "./results/train_wh/config1_preset_config1_epoch_20.pt"

if not os.path.exists('./results'):
  os.makedirs('./results/dry', exist_ok=True)
  os.makedirs('./results/wet', exist_ok=True)
  os.makedirs('./results/gt', exist_ok=True)

config_name = os.path.basename(config_path)[:-4]
preset_name = os.path.basename(preset_path)[:-4]

with open(preset_path, "r") as in_f:
    preset = yaml.safe_load(in_f)

# get preset value as tensor
preset_values = []
for _, value in preset.items():
    preset_values.append(value)

# # to match input shapes for FiLM
# preset_values = tr.tensor(preset_values).squeeze(-1)

with open(config_path, "r") as in_f:
    config = yaml.safe_load(in_f)

# get config dict
config_dict = {}
for k,v in config.items():
    config_dict[k] = v

# set config values
bs = config_dict['batch_size']
sr = config_dict['sr']
n_samples = config_dict['n_samples']
end_buffer_n_samples = config_dict['end_buffer_n_samples']
n_retries = config_dict['n_retries']

# random chunk can only support bs = 1, may change this in the future
assert bs == 1
preset_values = torch.tensor(preset_values)
preset_values_batched = preset_values.repeat((bs, 1))
preset_values_batched_unsqueezed = preset_values_batched

# overwrite "nparams" default config
n_retries = 2
config_dict['nparams'] = len(preset_values)

print("preset_values shape:", preset_values.shape)
print("config_dict:", config_dict)

# initialise processor and parameter network and W-H
wahwah = wah_wah(44100)
parameter_net = ParameterNetwork(preset_values_batched_unsqueezed.shape[-1], wahwah.num_params)
wh = Weiner_Hammerstein()
parameter_net.to(device)

# params stored as dict
_, flat_params_l1 = load_eq_parameters(l1_config_path)
_, flat_params_l2 = load_eq_parameters(l2_config_path)

# load model params
checkpoint = torch.load(model_path)
parameter_net.load_state_dict(checkpoint['model_state_dict'])
parameter_net.eval()

# a list of random indexes per file
list_of_permuted_indexes_test = []
file_list = sorted(os.listdir(base_path))

for i in range(len(file_list)):
    file_list[i] = os.path.join(base_path, file_list[i])

for input_path in file_list:
    if os.path.basename(input_path) == '.DS_Store':
        continue
    file_info = torchaudio.info(input_path)

    total_number_of_frames = file_info.num_frames
    # list
    random_indexes = random_permute(n_retries, total_number_of_frames, n_samples, end_buffer_n_samples, seed=34567)
    list_of_permuted_indexes_test.append(random_indexes)


print("list_of_permuted_indexes_test:", list_of_permuted_indexes_test)

# make dataloaders
test_dataset_dry = RandomAudioChunkDataset(test_input_paths_clean,
                                           list_of_permuted_indexes_test,
                                           n_samples,
                                           sr,
                                           n_retries=10,
                                           num_examples_per_epoch=100)

test_dataset_wet = RandomAudioChunkDataset(test_input_paths_wet,
                                           list_of_permuted_indexes_test,
                                           n_samples,
                                           sr,
                                           n_retries=10,
                                           num_examples_per_epoch=100)



test_dataloader_dry = torch.utils.data.DataLoader(test_dataset_dry, batch_size=bs)
test_dataloader_wet = torch.utils.data.DataLoader(test_dataset_wet, batch_size=bs)

# batch preset
preset_values_batched = preset_values.repeat((bs, 1))
preset_values_batched_unsqueezed = preset_values_batched.unsqueeze(1)
print("preset_values_batched_unsqueezed shape:", preset_values_batched_unsqueezed.shape)

# set up loss functions for evaluation
l1   = torch.nn.L1Loss()
stft = auraloss.freq.STFTLoss()
meter = pyln.Meter(44100)

loss_func = {'l1': l1, 'stft': stft, 'meter': meter}



