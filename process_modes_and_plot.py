import os
import torch
import torchaudio
import yaml
from tqdm import tqdm
from models.processors.trainable_parametric_eq import ParametricEQ
from models.processors.svf_biquads import StateVariableFilter
from models.processors.dasp_processor import wah_wah
from data_preprocess.random_chunk_preprocess import RandomAudioChunkDataset
from ..utils.plot_spectrogram import plot_spectrogram

# Device
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Paths and config
model_path = "./train_wh_graphic_eq_models_07-17/preset_config1_preset_config1_epoch_1_450.pt"
mount_point = "/mnt/gestalt/home/mku666"
test_input_paths_clean = os.path.join(mount_point, "wah_emulation/EGDB_test_subset/mono/dry")
test_input_paths_gt = os.path.join(mount_point, "wah_emulation/EGDB_test_subset/mono/wet")
config_path = "configs/config1.yml"
preset_path = "configs/preset_config1.yml"
sr = 44100
n_samples = 88200
bs = 1

# Output directories
out_base = "./results/mono"
mode_dirs = {
    "svf_only": os.path.join(out_base, "svf_only"),
    "pre_eq_svf": os.path.join(out_base, "pre_eq_svf"),
    "svf_post_eq": os.path.join(out_base, "svf_post_eq"),
}
for d in mode_dirs.values():
    os.makedirs(d, exist_ok=True)

# Load preset and config
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
preset = load_yaml(preset_path)
config = load_yaml(config_path)
preset_values = torch.tensor([v for v in preset.values()], dtype=torch.float32).to(DEVICE)
preset_values = preset_values.unsqueeze(0)  # shape (1, n_params)

# Data loader
dataset = RandomAudioChunkDataset(test_input_paths_clean, test_input_paths_gt, n_samples, sr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=2, pin_memory=True)

# Models
svf = StateVariableFilter(num_filters=1, device=DEVICE)
wahwah = wah_wah(sr, processor_fn=svf.svf_biquads_processor_forward)
trainable_eq_pre = ParametricEQ(sample_rate=sr).to(DEVICE)
trainable_eq_post = ParametricEQ(sample_rate=sr).to(DEVICE)

# Load weights
checkpoint = torch.load(model_path, map_location=DEVICE)
trainable_eq_pre.load_state_dict(checkpoint['graphic_eq_pre_state_dict'])
trainable_eq_post.load_state_dict(checkpoint['graphic_eq_post_state_dict'])

# Parameter network for DSP controls
from inference import ParameterNetwork
parameter_net = ParameterNetwork(preset_values.shape[-1], wahwah.num_params).to(DEVICE)
parameter_net.load_state_dict(checkpoint['parameter_net_state_dict'])
parameter_net.eval()
trainable_eq_pre.eval()
trainable_eq_post.eval()
# wahwah.eval()  # wah_wah is not a torch.nn.Module, so no eval()

# Processing functions
def process_svf_only(x, p_hat):
    return wahwah.process_normalized(x, p_hat)
def process_pre_eq_svf(x, p_hat):
    x = trainable_eq_pre(x)
    return wahwah.process_normalized(x, p_hat)
def process_svf_post_eq(x, p_hat):
    x = wahwah.process_normalized(x, p_hat)
    return trainable_eq_post(x)

# Main loop
for batch in tqdm(dataloader, desc="Processing audio"):
    (dry, dry_path, dry_idx), _ = batch
    dry = dry.to(DEVICE)
    dry = dry.view(-1, 1, dry.shape[-1])
    p_hat = parameter_net(preset_values)
    # Mode 1: SVF only
    wet1 = process_svf_only(dry, p_hat)
    # Mode 2: Pre-EQ + SVF
    wet2 = process_pre_eq_svf(dry, p_hat)
    # Mode 3: SVF + Post-EQ
    wet3 = process_svf_post_eq(dry, p_hat)
    # Save
    base = os.path.splitext(os.path.basename(dry_path[0]))[0]
    idx = dry_idx[0].item() if hasattr(dry_idx[0], 'item') else dry_idx[0]
    dry_out = os.path.join(out_base, f"dry_{base}_{idx}.wav")
    wet1_out = os.path.join(mode_dirs["svf_only"], f"wet_{base}_{idx}.wav")
    wet2_out = os.path.join(mode_dirs["pre_eq_svf"], f"wet_{base}_{idx}.wav")
    wet3_out = os.path.join(mode_dirs["svf_post_eq"], f"wet_{base}_{idx}.wav")
    torchaudio.save(dry_out, dry.cpu().squeeze(1), sr, backend="soundfile")
    torchaudio.save(wet1_out, wet1.cpu().squeeze(1), sr, backend="soundfile")
    torchaudio.save(wet2_out, wet2.cpu().squeeze(1), sr, backend="soundfile")
    torchaudio.save(wet3_out, wet3.cpu().squeeze(1), sr, backend="soundfile")
    # Plot spectrograms
    plot_spectrogram(dry_out, wet1_out, save_path=wet1_out.replace('.wav', '_spec.png'))
    plot_spectrogram(dry_out, wet2_out, save_path=wet2_out.replace('.wav', '_spec.png'))
    plot_spectrogram(dry_out, wet3_out, save_path=wet3_out.replace('.wav', '_spec.png')) 