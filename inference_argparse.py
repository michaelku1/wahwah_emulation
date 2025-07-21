"""
perform inference on mono/poly data
"""


import os
import sys

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import numpy as np

import pytorch_lightning as pl
import auraloss
import pyloudnorm as pyln

from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset, random_permute
from models.processors.trainable_parametric_eq import ParametricEQ
from models.processors.svf_biquads import StateVariableFilter
from models.processors.dasp_processor import wah_wah

from scipy import fftpack

import yaml
from typing import Dict, Optional, List, Any, Tuple, Type, Union
from collections import defaultdict
from tqdm import tqdm
import random
import argparse

global device

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True # enable appropriate cnn algorithm for the corresponding hardware

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(42)
seed = 42

print(f"using device: {device}")

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

        print(x.shape)

        # out x: (bs, sample_length)
        x = trainable_graphic_eq_pre(x)

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


def test_loop(test_dataloader,
              parameter_net,
              trainable_graphic_eq_pre,
              trainable_graphic_eq_post,
              wh,
              processor,
              criterion,
              config_dict,
              preset,
              config_name,
              preset_name):


    # for plotting
    test_loss_history = []

    # validation
    for batch_idx, pbar in enumerate(test_dataloader):
      with torch.no_grad():

        # returns: (tensor(), List[str], List[int])
        (dry_test, dry_test_file_path, dry_test_start_idx), (wet_test, wet_test_file_path, wet_test_start_idx) = pbar

        # check train data dry and wet
        assert os.path.basename(dry_test_file_path[0]) == os.path.basename(wet_test_file_path[0]), "dry and wet file path are not the same"
        assert dry_test_start_idx == wet_test_start_idx, "dry and wet start index are not the same"
        assert dry_test.shape == wet_test.shape, "dry and wet train data shape are not the same"

        test_x = dry_test
        test_y = wet_test

        # e.g 2s
        sample_length = test_x.shape[-1]

        dry_audio_name = os.path.basename(dry_test_file_path[0])[:-4]
        wet_audio_name = os.path.basename(wet_test_file_path[0])[:-4]
        gt_name = wet_audio_name # sake of readability

        # for sake of readability
        index = dry_test_start_idx

        # (bs, chunk_length, sample_length) --> (bs * chunk_length, sample_length)
        # test_x = test_x.squeeze(0)
        # test_y = test_y.squeeze(0)

        test_x = test_x.to(device)
        test_y = test_y.to(device)
        preset = preset.to(device)

        # ensure channel dim is one for shape compatibility with tcn model
        # assert test_x.shape[1] == 1 and test_y.shape[1] == 1

        # (1, bs, sample_length)
        # test_x = test_x.view(-1, test_x.shape[-1]).unsqueeze(0)
        # test_y = test_y.view(-1, test_y.shape[-1]).unsqueeze(0)

        test_x = test_x.view(-1, 1, test_x.shape[-1])
        test_y = test_y.view(-1, 1, test_y.shape[-1])

        # print("test shape:", test_x.shape)
        # print("test shape:", test_y.shape)

        # (1, 1, num_controls) --> (1, num_controls)
        # p_hat = parameter_net(preset).squeeze(0)
        p_hat = parameter_net(preset)
        # print("p_hat shape:", p_hat.shape)

        # (1, bs, sample_length)
        y_hat = wh(test_x, trainable_graphic_eq_pre, trainable_graphic_eq_post, processor, p_hat)

        # print("test_y shape", test_y.shape)
        # print("y_hat shape", y_hat.shape)

        test_loss = defaultdict(list)
        for name, loss_func in criterion.items():
          if name == "meter":
            # accepts 1D array
            target_lufs = loss_func.integrated_loudness(test_y.view(-1).cpu().numpy())
            output_lufs = loss_func.integrated_loudness(y_hat.reshape(-1).cpu().numpy())
            l1_lufs = np.abs(output_lufs - target_lufs)
            test_loss['l1_lufs'].append(l1_lufs)

          elif name == "l1":
            l1i_loss = (loss_func(y_hat, test_y)).cpu().numpy()
            test_loss['l1i_loss'].append(l1i_loss)

          elif name == "stft":
            # y_hat = y_hat.view(y_hat.shape[0], 1, y_hat.shape[-1])
            # test_y = test_y.view(test_y.shape[0], 1, test_y.shape[-1])
            stfti_loss = (loss_func(y_hat, test_y)).cpu().numpy()
            test_loss['stfti_loss'].append(stfti_loss)

        print(f"test losses: ", test_loss)
        test_loss_history.append(test_loss)

        # print(test_x.shape)
        # print(y_hat.shape)
        # print(test_y.shape)

        if not os.path.isdir(dry_audio_save_path):
          os.makedirs(dry_audio_save_path, exist_ok=True)

        if not os.path.isdir(wet_audio_save_path):
          os.makedirs(wet_audio_save_path, exist_ok=True)

        if not os.path.isdir(wet_audio_save_path):
          os.makedirs(gt_audio_save_path, exist_ok=True)

        # store corresponding n second chunk
        # dry audio: test_x
        # wet: y_hat
        # gt: test_y
        # print(test_x.shape)
        # print(test_y.shape)
        # print(y_hat.shape)

        # print(dry_audio_name)
        # print(wet_audio_name)
        # print(gt_name)

        test_x = test_x.view(-1, test_x.shape[-1]).cpu()
        test_y = test_y.view(-1, test_y.shape[-1]).cpu()
        # NOTE 
        y_hat = y_hat.view(-1, y_hat.shape[-1]).cpu()

        print("check test_x shape", test_x.shape)

        for i, v in enumerate(index):
          # (1, sample_length) (soundfile backend only accepts 2D tensors)
          test_x_seg = test_x[i, :].unsqueeze(0)
          y_hat_seg = y_hat[i, :].unsqueeze(0)
          test_y_seg = test_y[i, :].unsqueeze(0)

          print("test_x_seg shape", test_x_seg.shape)
          print("y_hat_seg shape", y_hat_seg.shape)
          print("test_y_seg shape", test_y_seg.shape)

          # torchaudio.save(os.path.join(dry_audio_save_path, f'{dry_audio_name}_{v.item()}.wav'), dry_, config_dict['sr'], backend="soundfile")
          # torchaudio.save(os.path.join(gt_audio_save_path, f'{gt_name}_{v.item()}.wav'), gt_, config_dict['sr'], backend="soundfile")
          # torchaudio.save(os.path.join(wet_audio_save_path, f'{wet_audio_name}_{v.item()}.wav'), wet_, config_dict['sr'], backend="soundfile")

          torchaudio.save(os.path.join(dry_audio_save_path, f'{dry_audio_name}_{v.item()}.wav'), test_x_seg, 44100, backend="soundfile")
          torchaudio.save(os.path.join(gt_audio_save_path, f'{gt_name}_{v.item()}.wav'), test_y_seg, 44100, backend="soundfile")
          torchaudio.save(os.path.join(wet_audio_save_path, f'{wet_audio_name}_{v.item()}.wav'), y_hat_seg, 44100, backend="soundfile")


############ specify data and config file paths ############

# saved model path
# model_path = "./train_wh_graphic_eq_models_07-17/preset_config1_preset_config1_epoch_1_450.pt"
# model_path = "./train_wh_graphic_eq_models_07-17/preset_config1_preset_config1_epoch_1_450.pt"
# mount_point = "/mnt/gestalt/home/mku666"
# test_input_paths_clean = "wah_emulation/EGDB_seen/EGDB_NA_Wah_valid_dry"
# test_input_paths_gt = "wah_emulation/EGDB_seen/EGDB_NA_Wah_valid_gt"


# test on mono data
# test_input_paths_clean = "wah_emulation/EGDB_test_subset/mono/dry"
# test_input_paths_gt = "wah_emulation/EGDB_test_subset/mono/wet"

# test on poly data
# test_input_paths_clean = "wah_emulation/EGDB_test_subset/poly/dry"
# test_input_paths_gt = "wah_emulation/EGDB_test_subset/poly/gt"

# config_path = "configs/config1.yml"
# preset_path = "configs/preset_config1.yml"

# # selected mono poly test set
# test_input_paths_clean = os.path.join(mount_point, test_input_paths_clean)
# test_input_paths_gt = os.path.join(mount_point, test_input_paths_gt)

# print("dry path:", test_input_paths_clean)
# print("gt path:", test_input_paths_gt)

# assert os.path.exists(test_input_paths_clean), "dry path does not exist"
# assert os.path.exists(test_input_paths_gt), "gt path does not exist"

### multi data
# test_input_paths_clean = "./dataset1/test/dry/"""
# test_input_paths_wet = "./dataset1/test/wet/"
# mount_point = "./dataset1/test/dry" # for obtaining file metadata info
# config_path = "./configs/config1.yml"
# preset_path = "./dataset1/preset_config1.yml"


### single data seen
# test_input_paths_clean = "./single_data/test/seen/dry/"
# test_input_paths_wet = "./single_data/test/seen/wet/"
# mount_point = "./single_data/test/seen/dry" # for obtaining file metadata info
# config_path = "./configs/config1.yml"
# preset_path = "./single_data/preset_config1.yml"

### single data unseen
# test_input_paths_clean = "./single_data/test/unseen/dry/"
# test_input_paths_wet = "./single_data/test/unseen/wet/"
# mount_point = "./single_data/test/unseen/dry" # for obtaining file metadata info
# config_path = "./configs/config1.yml"
# preset_path = "./single_data/preset_config1.yml"

############ specify save paths ############
global dry_audio_save_path
global wet_audio_save_path
global gt_audio_save_path

# trained_model_version_date = model_path.split('_')[-1]

# dry_audio_save_path = f'./results/test_single/mono/{trained_model_version_date}/dry'
# wet_audio_save_path = f'./results/test_single/mono/{trained_model_version_date}/wet'
# gt_audio_save_path = f'./results/test_single/mono/{trained_model_version_date}/gt/'

# # saved audio paths
# if not os.path.exists(dry_audio_save_path):
#     os.makedirs(dry_audio_save_path, exist_ok=True)

# if not os.path.exists(wet_audio_save_path):
#     os.makedirs(wet_audio_save_path, exist_ok=True)

# if not os.path.exists(gt_audio_save_path):
#     os.makedirs(gt_audio_save_path, exist_ok=True)

# config_name = os.path.basename(config_path)[:-4]
# preset_name = os.path.basename(preset_path)[:-4]

# with open(preset_path, "r") as in_f:
#     preset = yaml.safe_load(in_f)

# # get preset value as tensor
# preset_values = []
# for _, value in preset.items():
#     preset_values.append(value)
# print("preset_values:", preset_values)
# # print("preset_values shape:", preset_values.shape)

# # to match input shapes for FiLM
# preset_values = tr.tensor(preset_values).squeeze(-1)

# with open(config_path, "r") as in_f:
#     config = yaml.safe_load(in_f)

# # get config dict
# config_dict = {}
# for k,v in config.items():
#     config_dict[k] = v

# # set config values
# # bs = config_dict['batch_size']
# # sr = config_dict['sr']
# # n_samples = config_dict['n_samples']
# # end_buffer_n_samples = config_dict['end_buffer_n_samples']
# # n_retries = config_dict['n_retries']
# # # overwrite "nparams" default config
# # config_dict['nparams'] = len(preset_values)

# silence_threshold_energy = 1e-6 # allowable silence threshold
# silence_fraction_allowed = 0.8 # factor of number of samples allowable for silence
# n_retries = 10 # number of random chunks to sample (treated as batch size in this context)
# end_buffer_n_samples = 0 # always zero
# n_samples = 88200
# sr = 44100

# num_examples_per_epoch=1000
# bs = 1 # set to 1 for now
# num_epochs = 100
# config_dict = None # will remove this later

# print("config:", config_dict)

# # for test data, sample chunks with as little silece parts as possible
# dataset_test = RandomAudioChunkDataset(test_input_paths_clean, test_input_paths_gt, n_samples, sr, silence_fraction_allowed=silence_fraction_allowed, silence_threshold_energy=silence_threshold_energy, n_retries=n_retries, end_buffer_n_samples=end_buffer_n_samples)
# test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=bs, num_workers=2, pin_memory=True)

# # random chunk can only support bs = 1, may change this in the future
# assert bs == 1

# # reshape preset input shape to fit model
# preset_values = torch.tensor(preset_values)
# preset_values_batched = preset_values.repeat((bs, 1))
# preset_values_batched_unsqueezed = preset_values_batched
# preset_values_batched_unsqueezed = preset_values_batched_unsqueezed.to(device)

# # initialise svf biquad
# svf = StateVariableFilter(num_filters=1, device=device)

# # initialise processor and parameter network and W-H
# wahwah = wah_wah(sr, processor_fn=svf.svf_biquads_processor_forward)
# num_taps = 1025
# window_size = 256

# # check processing function type
# print("wahwah process function type:", wahwah.process_fn)

# parameter_net = ParameterNetwork(preset_values_batched_unsqueezed.shape[-1], wahwah.num_params) # parameter network mapping to dsp parameters
# trainable_graphic_eq_pre = ParametricEQ(sample_rate=sr) # pre-emphasis filter
# trainable_graphic_eq_post = ParametricEQ(sample_rate=sr)# post-emphasis filter
# wh = Weiner_Hammerstein()

# # move models to device
# parameter_net = parameter_net.eval()
# trainable_graphic_eq_pre.eval()
# trainable_graphic_eq_post.eval()
# wh.eval()

# trainable_graphic_eq_pre = trainable_graphic_eq_pre.to(device)
# trainable_graphic_eq_pre = trainable_graphic_eq_post.to(device)
# parameter_net = parameter_net.to(device)
# wh = wh.to(device)

# # TensorBoard writer
# writer = SummaryWriter(log_dir='train_wh_graphic_eq_runs/experiment_1')

# # load model params
# checkpoint = torch.load(model_path)
# parameter_net.load_state_dict(checkpoint['parameter_net_state_dict'])
# trainable_graphic_eq_pre.load_state_dict(checkpoint['graphic_eq_pre_state_dict'])
# trainable_graphic_eq_post.load_state_dict(checkpoint['graphic_eq_post_state_dict'])

# # set up loss functions for evaluation
# l1   = torch.nn.L1Loss()
# stft = auraloss.freq.STFTLoss()
# meter = pyln.Meter(44100)

# loss_func = {'l1': l1, 'stft': stft, 'meter': meter}

# test_loop(test_dataloader,
#           parameter_net,
#           wh,
#           wahwah,
#           loss_func,
#           config_dict,
#           preset_values_batched_unsqueezed,
#           config_name,
#           preset_name)

def main():
    parser = argparse.ArgumentParser(description="Perform inference on mono or poly data with a specified model.")
    parser.add_argument('--data_type', type=str, choices=['mono', 'poly'], required=True, help='Type of data to perform inference on: mono or poly')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pt)')
    parser.add_argument('--mount_point', type=str, default='/mnt/gestalt/home/mku666', help='Base directory for data (default: /mnt/gestalt/home/mku666)')
    parser.add_argument('--config_path', type=str, default='configs/config1.yml', help='Path to config YAML file')
    parser.add_argument('--preset_path', type=str, default='configs/preset_config1.yml', help='Path to preset YAML file')
    args = parser.parse_args()

    # Set paths based on data_type
    if args.data_type == 'mono':
        test_input_paths_clean = 'wah_emulation/EGDB_test_subset/mono/dry'
        test_input_paths_gt = 'wah_emulation/EGDB_test_subset/mono/wet'
    else:
        test_input_paths_clean = 'wah_emulation/EGDB_test_subset/poly/dry'
        test_input_paths_gt = 'wah_emulation/EGDB_test_subset/poly/gt'


    mount_point = args.mount_point
    test_input_paths_model = args.model_path
    config_path = args.config_path
    preset_path = args.preset_path

    # Compose full paths
    test_input_paths_clean = os.path.join(mount_point, test_input_paths_clean)
    test_input_paths_gt = os.path.join(mount_point, test_input_paths_gt)
    # test_input_paths_model = os.path.join(mount_point, test_input_paths_model)

    print("dry path:", test_input_paths_clean)
    print("gt path:", test_input_paths_gt)
    print("model path:", test_input_paths_model)

    assert os.path.exists(test_input_paths_clean), "dry path does not exist"
    assert os.path.exists(test_input_paths_gt), "gt path does not exist"
    assert os.path.exists(test_input_paths_model), "model path does not exist"

    config_name = os.path.basename(config_path)[:-4]
    preset_name = os.path.basename(preset_path)[:-4]

    with open(preset_path, "r") as in_f:
        preset = yaml.safe_load(in_f)

    # get preset value as tensor
    preset_values = []
    for _, value in preset.items():
        preset_values.append(value)
    print("preset_values:", preset_values)

    with open(config_path, "r") as in_f:
        config = yaml.safe_load(in_f)

    config_dict = {}
    for k, v in config.items():
        config_dict[k] = v

    silence_threshold_energy = 1e-6
    silence_fraction_allowed = 0.8
    n_retries = 10
    end_buffer_n_samples = 0
    n_samples = 88200
    sr = 44100

    num_examples_per_epoch = 1000
    bs = 1
    num_epochs = 100
    config_dict = None

    print("config:", config_dict)

    dataset_test = RandomAudioChunkDataset(test_input_paths_clean, test_input_paths_gt, n_samples, sr, silence_fraction_allowed=silence_fraction_allowed, silence_threshold_energy=silence_threshold_energy, n_retries=n_retries, end_buffer_n_samples=end_buffer_n_samples)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=bs, num_workers=2, pin_memory=True)

    assert bs == 1

    preset_values = torch.tensor(preset_values)
    preset_values_batched = preset_values.repeat((bs, 1))
    preset_values_batched_unsqueezed = preset_values_batched
    preset_values_batched_unsqueezed = preset_values_batched_unsqueezed.to(device)

    svf = StateVariableFilter(num_filters=1, device=device)
    wahwah = wah_wah(sr, processor_fn=svf.svf_biquads_processor_forward)
    num_taps = 1025
    window_size = 256

    print("wahwah process function type:", wahwah.process_fn)

    parameter_net = ParameterNetwork(preset_values_batched_unsqueezed.shape[-1], wahwah.num_params)
    trainable_graphic_eq_pre = ParametricEQ(sample_rate=sr)
    trainable_graphic_eq_post = ParametricEQ(sample_rate=sr)
    wh = Weiner_Hammerstein()

    parameter_net = parameter_net.eval()
    trainable_graphic_eq_pre.eval()
    trainable_graphic_eq_post.eval()
    wh.eval()

    trainable_graphic_eq_pre = trainable_graphic_eq_pre.to(device)
    trainable_graphic_eq_pre = trainable_graphic_eq_post.to(device)
    parameter_net = parameter_net.to(device)
    wh = wh.to(device)

    writer = SummaryWriter(log_dir='train_wh_graphic_eq_runs/experiment_1')

    checkpoint = torch.load(test_input_paths_model)
    parameter_net.load_state_dict(checkpoint['parameter_net_state_dict'])
    trainable_graphic_eq_pre.load_state_dict(checkpoint['graphic_eq_pre_state_dict'])
    trainable_graphic_eq_post.load_state_dict(checkpoint['graphic_eq_post_state_dict'])

    l1 = torch.nn.L1Loss()
    stft = auraloss.freq.STFTLoss()
    meter = pyln.Meter(44100)

    loss_func = {'l1': l1, 'stft': stft, 'meter': meter}

    # Set up save paths
    global dry_audio_save_path
    global wet_audio_save_path
    global gt_audio_save_path
    trained_model_version_date = os.path.basename(test_input_paths_model).split('_')[-1].split('.')[0]
    dry_audio_save_path = f'./results/test_single/{args.data_type}/{trained_model_version_date}/dry'
    wet_audio_save_path = f'./results/test_single/{args.data_type}/{trained_model_version_date}/wet'
    gt_audio_save_path = f'./results/test_single/{args.data_type}/{trained_model_version_date}/gt/'

    if not os.path.exists(dry_audio_save_path):
        os.makedirs(dry_audio_save_path, exist_ok=True)
    if not os.path.exists(wet_audio_save_path):
        os.makedirs(wet_audio_save_path, exist_ok=True)
    if not os.path.exists(gt_audio_save_path):
        os.makedirs(gt_audio_save_path, exist_ok=True)

    test_loop(test_dataloader,
              parameter_net,
              trainable_graphic_eq_pre,
              trainable_graphic_eq_post,
              wh,
              wahwah,
              loss_func,
              config_dict,
              preset_values_batched_unsqueezed,
              config_name,
              preset_name)

if __name__ == '__main__':
    main()