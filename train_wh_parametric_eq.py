
"""
Linear filters (following linear, non-linear and linear filter architecture) for W-H implementation with FIR (frequency impulse response),
this is implemented according to the "Reverse Engineering Memoryless Distortion Effects with
Differentiable Waveshapers" paper.

parametric EQ (L1) --> NN --> wah fx processor (N) --> parametric EQ (L2)

"""
"""
Linear filters (following linear, non-linear and linear filter architecture) for W-H implementation with FIR (frequency impulse response),
this is implemented according to the "Reverse Engineering Memoryless Distortion Effects with
Differentiable Waveshapers" paper.

parametric EQ (L1) --> NN --> wah fx processor (N) --> parametric EQ (L2)

"""
import os
import sys
from datetime import datetime

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import numpy as np

import pytorch_lightning as pl
import auraloss

from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset, random_permute
from models.processors.trainable_parametric_eq import ParametricEQ
from models.processors.svf_biquads import StateVariableFilter
from models.processors.dasp_processor import wah_wah
from scipy import fftpack

import yaml
from typing import Dict, Optional, List, Any, Tuple, Type, Union
from tqdm import tqdm

global device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True # enable appropriate cnn algorithm for the corresponding hardware

import random
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(42)
seed = 42

print(f"using device: {device}")

def compute_mean_loss(loss_history):
    """
    Compute mean loss for each type of loss.

    Args:
        loss_history: list of dicts, each dict contains loss values per batch.
    Returns:
        mean_loss_history: dict mapping each loss name to its average value.
    """
    mean_loss_history = {}
    for loss_dict in loss_history:
        for name, value in loss_dict.items():
            mean_loss_history.setdefault(name, []).append(value)

    for name, values in mean_loss_history.items():
        mean_loss_history[name] = sum(values) / len(values)

    return mean_loss_history


def train_loop(num_epochs,
               train_dataloader,
               valid_dataloader,
               parameter_net,
               wh,
               processor,
               optimizer,
               criterion,
               config_dict,
               preset,
               config_name,
               preset_name):

      global seed
      global last_epoch
      global last_iter

      number_epochs_to_run = num_epochs

      if last_epoch:
        number_epochs_to_run = num_epochs - last_epoch
        print(f'starting from model {pretrained_model_path}, with epoch number {last_epoch}')
      else:
        last_epoch = 0
        number_epochs_to_run = num_epochs
        print(f'starting from epoch 1')

      if last_iter != 0:
        number_iters_to_run = num_examples_per_epoch - last_iter
        print(f'starting from model {pretrained_model_path}, with starting iter number {last_iter}, and {number_iters_to_run} steps remaining')

      else:
        last_iter = 0
        number_iters_to_run = num_examples_per_epoch
        print(f'starting iter 1, train for {number_iters_to_run} steps')

      for epoch_num in range(number_epochs_to_run):
          train_loss_history = [] # of length equal to number of iterations, tracking batch losses for each epoch, stores dicts
          valid_loss_history = [] # of length equal to number of iterations, tracking batch losses for each epoch, stores dicts

          for iter_num in tqdm(range(number_iters_to_run)):
              seed += 1
              random_indexes_train = dataset_train.get_permuted_indexes(seed)
              dataset_train.set_random_indexes(random_indexes_train)

              # NOTE only use this when __len__ = 1 (overfitting on on batch and first batch only)
              train_data = next(iter(train_dataloader))
              valid_data = next(iter(valid_dataloader))

              # NOTE this is the only place where we use the dataloader
              # for batch, pbar in tqdm(enumerate(zip(dataloader_train, dataloader_valid))):
              #     train_data, valid_data = pbar

              # NOTE everything should be of size==batch
              (dry_train, dry_train_file_path, dry_train_start_idx), (wet_train, wet_train_file_path, wet_train_start_idx) = train_data
              (dry_valid, dry_valid_file_path, dry_valid_start_idx), (wet_valid, wet_valid_file_path, wet_valid_start_idx) = valid_data

              # check train data dry and wet
              assert os.path.basename(dry_train_file_path[0]) == os.path.basename(wet_train_file_path[0]), "dry and wet file path are not the same"
              assert dry_train_start_idx == wet_train_start_idx, "dry and wet start index are not the same"
              assert dry_train.shape == wet_train.shape, "dry and wet train data shape are not the same"

              # check valid data dry and wet
              assert os.path.basename(dry_valid_file_path[0]) == os.path.basename(wet_valid_file_path[0]), "dry and wet file path are not the same"
              assert dry_valid_start_idx == wet_valid_start_idx, "dry and wet start index are not the same"
              assert dry_valid.shape == wet_valid.shape, "dry and wet valid data shape are not the same"

              # bs here is the bs defined by dataloader, whereas chunk num is after random sampling from the dataloader
              # consider fix this logic later
              # (bs, chunk_num, sample_length) --> (bs * chunk_num, sample_length) --> (bs, 1, sample_length)
              train_x = dry_train.view(-1, 1, dry_train.shape[-1]) # (bs, 1, sample_length)
              train_y = wet_train.view(-1, 1, wet_train.shape[-1]) # (bs, 1, sample_length)
              valid_x = dry_valid.view(-1, 1, dry_valid.shape[-1])
              valid_y = wet_valid.view(-1, 1, wet_valid.shape[-1])

              # send data to device
              train_x = train_x.to(device)
              train_y = train_y.to(device)
              valid_x = valid_x.to(device)
              valid_y = valid_y.to(device)
              preset = preset.to(device)

              # print(train_x.shape)
              # print(train_y.shape)
              # print(valid_x.shape)
              # print(valid_y.shape)
              # print(preset.shape)

              # (1, bs, sample_length)
              assert train_x.shape == train_y.shape
              assert valid_x.shape == valid_y.shape

              ################ train #################
              # wh model forward pass
              p_hat = parameter_net(preset)
              y_hat_train = wh(train_x, trainable_graphic_eq_pre, trainable_graphic_eq_post, wahwah, p_hat)

              train_total_losses = 0 # batch gradient descent, so loss will be zeroed every time it is updated
              loss_info = {}
              for loss_name, loss_func in criterion.items():
                      loss = loss_func(y_hat_train, train_y)
                      train_total_losses += loss
                      loss_info[loss_name] = loss.detach().cpu().item() # record each loss item, detach from graph
              loss_info['total_losses'] = train_total_losses.detach().cpu().item()
              train_loss_history.append(loss_info)

              # print("after detach", train_total_losses.requires_grad)

              optimizer.zero_grad()
              train_total_losses.backward()
              optimizer.step()

              cur_iter_num = iter_num + last_iter
              cur_epoch_num = epoch_num + last_epoch

              print(f"training losses on iter {cur_iter_num}", loss_info)

              ################ validation #################
              with torch.no_grad():
                p_hat = parameter_net(preset)
                y_hat_valid = wh(valid_x, trainable_graphic_eq_pre, trainable_graphic_eq_post, wahwah, p_hat)

                valid_total_losses = 0 # batch gradient descent, so loss will be zeroed every time it is updated
                loss_info = {}
                for loss_name, loss_func in criterion.items():
                        loss = loss_func(y_hat_valid, valid_y)
                        valid_total_losses += loss
                        loss_info[loss_name] = loss.detach().cpu().item() # record each loss item
                loss_info['total_losses'] = valid_total_losses.detach().cpu().item()
                valid_loss_history.append(loss_info)

              print(f"validation losses on iter {cur_iter_num}: ", loss_info)


              ################ save model for every N iterations ################
              if iter_num % 50 == 0:
                  model_save_path = os.path.join(output_dir, config_name + '_' + preset_name + '_' + f"epoch_{cur_epoch_num}_{cur_iter_num}" + '.pt')

                  # make model directory
                  if os.path.exists(f"{output_dir}"):
                    pass
                  else:
                    os.makedirs(f"{output_dir}")

                  # save model per epoch
                  torch.save({
                  'iter_num': cur_iter_num,
                  'epoch': cur_epoch_num,
                  'parameter_net_state_dict': parameter_net.state_dict(),
                  'graphic_eq_pre_state_dict': trainable_graphic_eq_pre.state_dict(),
                  'graphic_eq_post_state_dict': trainable_graphic_eq_post.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, model_save_path)


          ################ end of one epoch ################
          # clear cache
          torch.cuda.empty_cache()

          # compute mean loss
          train_loss_history = compute_mean_loss(train_loss_history)
          valid_loss_history = compute_mean_loss(valid_loss_history)

          # show training loss every epoch
          train_loss_history_formatted = ", ".join(f"{k}: {v:.4e}" for k, v in train_loss_history.items())
          print(f"train losses: {train_loss_history_formatted}")
          for loss_name, loss_value in train_loss_history.items():
              writer.add_scalar(f"{output_dir}/train_loss/train/{loss_name}", loss_value, epoch_num)


          # show validation loss every epoch
          valid_loss_history_formatted = ", ".join(f"{k}: {v:.4e}" for k, v in valid_loss_history.items())
          print(f"valid losses: {valid_loss_history_formatted}")
          for loss_name, loss_value in valid_loss_history.items():
              writer.add_scalar(f"{output_dir}/valid_loss/train/{loss_name}", loss_value, epoch_num)


          model_save_path = os.path.join(output_dir, config_name + '_' + preset_name + '_' + f"epoch_{epoch_num}" + '.pt')

          # make model directory
          if os.path.exists(f"{output_dir}"):
            pass
          else:
            os.makedirs(f"{output_dir}")


          # save model per epoch
          torch.save({
          'epoch': cur_epoch_num,
          'parameter_net_state_dict': parameter_net.state_dict(),
          'graphic_eq_pre_state_dict': trainable_graphic_eq_pre.state_dict(),
          'graphic_eq_post_state_dict': trainable_graphic_eq_post.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          }, model_save_path)

          # make sure memory is not accumulated
          del train_loss_history
          del valid_loss_history

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

    def forward(self, x, trainable_graphic_eq_pre, trainable_graphic_eq_post, fx_processor, p_hat, sr=44100):
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

        # out x: (bs, sample_length)
        x = trainable_graphic_eq_pre(x)

        print(x.shape)

        # BUG when n_retires = 1, this breaks the shape, need to consider this edge case
        # out x: (bs, sample_length)
        x = fx_processor.process_normalized(x.view(-1, x.shape[-1]), p_hat)

        # out x: (bs, 1, sample_length)
        x = x.view(x.shape[0], 1, x.shape[-1])

        # out x: (bs, sample_length)
        x = trainable_graphic_eq_post(x)

        # out x: (1, bs, sample_length)
        # x = x.transpose(0,1)
        # BUG shape inconsistent to parametric implementation, needs checking
        x = x.view(x.shape[0], 1, x.shape[-1])

        # debug
        # x = x*torch.tensor([1.], device=device, requires_grad=True)

        return x


# Get today's date
today = datetime.today()
date_str = today.strftime("%m-%d")

# dataset, config paths

# dataset, config paths
global output_dir

base_path = "./"
# data_run_type = 'single_data'
# data_run_type = 'all_data'
# model_run_type = "xxx_models"
model_run_type = "train_wh_graphic_eq_models"
store_model_path_name = f"{model_run_type}_{date_str}"

# pretrained_model_path = "./single_data/single_data_train_wh_graphic_eq_models_06-19/config1_preset_config1_epoch_20.pt"
# pretrained_model_path = "./single_data_train_wh_graphic_eq_models_07-01/preset_config1_preset_config1_epoch_1_440.pt"
# pretrained_model_path = "./train_wh_graphic_eq_models_07-03/preset_config1_preset_config1_epoch_1_500.pt"
pretrained_model_path = "./train_wh_graphic_eq_models_07-04/preset_config1_preset_config1_epoch_5_953.pt"
continual_training = False

if continual_training:
  assert os.path.exists(pretrained_model_path), "pretrained model path does not exist, please provide pretrained model path"
  print("pretrained model path name: ", pretrained_model_path)


preset_path = "./configs/preset_config1.yml"
config_name = "preset_config1"

# 10 min training audio,
# train_input_paths_clean = "./dataset1/train/dry"
# train_input_paths_wet = "./dataset1/train/wet"
# valid_input_paths_clean = "./dataset1/validation/seen/dry"
# valid_input_paths_wet = "./dataset1/validation/seen/wet"
# valid_input_paths_clean = "./dataset1/validation/unseen/dry"
# valid_input_paths_wet = "./dataset1/validation/unseen/wet"

# overfitting single data
# train_input_paths_clean = "./single_data/train/dry"
# train_input_paths_wet = "./single_data/train/wet"
# valid_input_paths_clean = "./single_data/validation/dry"
# valid_input_paths_wet = "./single_data/validation/wet"

mount_point = "/mnt/gestalt/home/mku666"

train_input_paths_clean = os.path.join(mount_point, "EGDB_DI")
train_input_paths_wet = os.path.join(mount_point, "NA_Wah/NA_WahFilter_7.5_Power_True_Bypass_False")
# train_input_paths_wet = os.path.join(mount_point, "NA_wah_75/poly/NA_WahFilter_7.5_Power_True_Bypass_False")
valid_input_paths_clean = os.path.join(mount_point, "EGDB_seen/EGDB_DI_train_dry")
valid_input_paths_wet = os.path.join(mount_point, "EGDB_seen/EGDB_DI_train_wet")
# valid_input_paths_clean = os.path.join(mount_point, "unseen/EGDB_DI_valid_dry")
# valid_input_paths_wet = os.path.join(mount_point, "unseen/EGDB_DI_valid_wet")

assert os.path.isdir(train_input_paths_clean)
assert os.path.isdir(train_input_paths_wet)
assert os.path.isdir(valid_input_paths_clean)
assert os.path.isdir(valid_input_paths_wet)

# params stored as dict
# _, flat_params = load_eq_parameters(config_path_eq)

# Create a directory based on that name
output_dir = os.path.join(base_path, store_model_path_name)
os.makedirs(output_dir, exist_ok=True)

# with open(config_path, "r") as in_f:
#     config = yaml.safe_load(in_f)

# # get config dict
# config_dict = {}
# for k,v in config.items():
#     config_dict[k] = v

with open(preset_path, "r") as in_f:
    preset = yaml.safe_load(in_f)

# check config parameters to be loaded

# config_name = os.path.basename(config_path)[:-4]
preset_name = os.path.basename(preset_path)[:-4]

# Create a directory based on that name
output_dir = os.path.join(base_path, store_model_path_name)
os.makedirs(output_dir, exist_ok=True)

# with open(config_path, "r") as in_f:
#     config = yaml.safe_load(in_f)

# get config dict
# config_dict = {}
# for k,v in config.items():
#     config_dict[k] = v

with open(preset_path, "r") as in_f:
    preset = yaml.safe_load(in_f)

# get preset value as tensor
preset_values = []
for _, value in preset.items():
    preset_values.append(value)

# to match input shapes for FiLM
preset_values = torch.tensor(preset_values).squeeze(-1)

# bs = config_dict['batch_size']
# sr = config_dict['sr']
# n_samples = config_dict['n_samples']
# end_buffer_n_samples = config_dict['end_buffer_n_samples']
# n_retries = config_dict['n_retries']

# overwrite "nparams" default config
# config_dict['nparams'] = len(preset_values)
print("preset values shape:", preset_values.shape)
print("preset values:", preset_values)
# print("config settings:", config_dict)
print("model output directory:", output_dir)


# these will later be moved to config
silence_threshold_energy = 1e-6 # allowable silence threshold
silence_fraction_allowed = 0.8 # factor of number of samples allowable for silence
n_retries = 32 # number of random chunks to sample (treated as batch size in this context)
end_buffer_n_samples = 0 # always zero
n_samples = 88200
sr = 44100

num_examples_per_epoch=1000
batch_size = 1 # set to 1 for now
num_epochs = 100
config_dict = None # will remove this later

dataset_train = RandomAudioChunkDataset(train_input_paths_clean,
                                        train_input_paths_wet,
                                        n_samples,
                                        num_examples_per_epoch=num_examples_per_epoch,
                                        sr=sr,
                                        silence_fraction_allowed=silence_fraction_allowed,
                                        silence_threshold_energy=silence_threshold_energy,
                                        n_retries=n_retries,
                                        end_buffer_n_samples=end_buffer_n_samples,
                                        should_peak_norm = True,
                                        seed=12345)

dataset_valid = RandomAudioChunkDataset(valid_input_paths_clean,
                                        valid_input_paths_wet,
                                        n_samples,
                                        num_examples_per_epoch=num_examples_per_epoch,
                                        sr=sr,
                                        silence_fraction_allowed=silence_fraction_allowed,
                                        silence_threshold_energy=silence_threshold_energy,
                                        n_retries=n_retries,
                                        end_buffer_n_samples=end_buffer_n_samples,
                                        should_peak_norm = True,
                                        seed=23456)

# batch_size=1 means sampling (n_retries, sample_length,)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=2, pin_memory=True)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, num_workers=2, pin_memory=True)

# get preset value as tensor
preset_values = []
for _, value in preset.items():
    preset_values.append(value)

preset_values = torch.tensor(preset_values)
preset_values_batched = preset_values.repeat((batch_size, 1))
preset_values_batched_unsqueezed = preset_values_batched
preset_values_batched_unsqueezed = preset_values_batched_unsqueezed.to(device)

# initialise processor and parameter network and W-H
wahwah = wah_wah(sr, processor_fn=StateVariableFilter.svf_biquads_processor_forward)
num_taps = 1025
window_size = 256

exit()

# check processing function type
print("wahwah process function type:", wahwah.process_fn)

parameter_net = ParameterNetwork(preset_values_batched_unsqueezed.shape[-1], wahwah.num_params) # parameter network mapping to dsp parameters
trainable_graphic_eq_pre = ParametricEQ(num_taps, window_size=window_size, fs=44100) # pre-emphasis filter
trainable_graphic_eq_post = ParametricEQ(num_taps, window_size=window_size, fs=44100)# post-emphasis filter
wh = Weiner_Hammerstein()

# move models to device
wh = wh.to(device)
parameter_net = parameter_net.to(device)
trainable_graphic_eq_pre.to(device)
trainable_graphic_eq_post.to(device)

# Checkpoint param group count
# print(f"Checkpoint param group count: {len(checkpoint['optimizer_state_dict']['param_groups'])}")

if continual_training and pretrained_model_path:
    print("loading pretrained model...")
    checkpoint = torch.load(pretrained_model_path)

    last_iter = checkpoint.get('iter_num', 0)
    last_epoch = checkpoint.get('epoch', 0)

    parameter_net.load_state_dict(checkpoint['parameter_net_state_dict'])
    trainable_graphic_eq_pre.load_state_dict(checkpoint['graphic_eq_pre_state_dict'])
    trainable_graphic_eq_post.load_state_dict(checkpoint['graphic_eq_post_state_dict'])

    # checkpoint values (e.g lr) are reserved
    optimizer = torch.optim.Adam([
        {'params': parameter_net.parameters()},
        {'params': trainable_graphic_eq_pre.parameters()},
        {'params': trainable_graphic_eq_post.parameters()},
    ])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for i, group in enumerate(optimizer.param_groups):
      print(f"Group {i} learning rate: {group['lr']}")

    print("success!")
else:
    last_epoch = 0
    optimizer = torch.optim.Adam([
        {'params': parameter_net.parameters(), 'lr': 1e-3},
        {'params': trainable_graphic_eq_pre.parameters(), 'lr': 1e-3},
        {'params': trainable_graphic_eq_post.parameters(), 'lr': 1e-3},
    ])

criterion = {"L1": nn.L1Loss(), "stft": auraloss.freq.STFTLoss()}

# TensorBoard writer
writer = SummaryWriter(log_dir=f'{output_dir}_runs/experiment_1')

# debug
seed = 42
# test random indexes
for _ in range(num_epochs):
    for _ in range(num_examples_per_epoch):
        seed += 1
        random_indexes_train = dataset_train.get_permuted_indexes(seed)
        dataset_train.set_random_indexes(random_indexes_train)

        # NOTE this is the only place where we use the dataloader
        train_data = next(iter(dataloader_train))
        valid_data = next(iter(dataloader_valid))
        
    # NOTE this is the only place where we use the dataloader
    # for batch, pbar in tqdm(enumerate(zip(dataloader_train, dataloader_valid))):
    #     train_data, valid_data = pbar

        # NOTE everything should be of size==batch 
        (dry_train, dry_train_file_path, dry_train_start_idx), (wet_train, wet_train_file_path, wet_train_start_idx) = train_data
        (dry_valid, dry_valid_file_path, dry_valid_start_idx), (wet_valid, wet_valid_file_path, wet_valid_start_idx) = valid_data

        # breakpoint()


# check input data shapes
# (dry_train, dry_train_file_path, dry_train_start_idx), (wet_train, wet_train_file_path, wet_train_start_idx) = data_train
# (dry_valid, dry_valid_file_path, dry_valid_start_idx), (wet_valid, wet_valid_file_path, wet_valid_start_idx) = data_valid
# print(dry_train.shape)
# print(dry_valid.shape)

# train_loop(num_epochs,
#         last_epoch,
#         dataloader_train,
#         dataloader_valid,
#         parameter_net,
#         wh,
#         wahwah,
#         optimizer,
#         criterion,
#         config_dict,
#         preset_values_batched_unsqueezed,
#         config_name,
#         preset_name)