import torchaudio
from random_chunk_preprocess import RandomAudioChunkDataset

import torch    
from tqdm import tqdm
import random
import numpy as np
import os

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_global_seed(42)

# file_path = "/Users/michael/Desktop/guitar data/EGDB subset/train/238.wav"
# index = 1563434
# n_samples = 44100

# num_frames = torchaudio.info(file_path).num_frames
# print(num_frames>index)

# audio_chunk, sr = torchaudio.load(file_path, frame_offset=index, num_frames=n_samples,)
# print(audio_chunk)
# print(audio_chunk.shape)
# print(sr)

def randint(low: int, high: int, n: int = 1):
    x = torch.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x

# test random permute
def random_permute(n_retries, file_n_samples, n_samples, end_buffer_n_samples, seed=12345):
    '''
    randomly permute index for random chunking
    '''
    
    torch.manual_seed(seed)

    # assuming input audio data is >> user defined audio chunk (e.g ~2s)
    high = int(file_n_samples - n_samples - end_buffer_n_samples + 1)
    random_indexes = []
    for _ in range(n_retries):
        start_idx = randint(0, high)
        random_indexes.append(start_idx)

    return random_indexes

input_paths_clean_train = "/Users/michael/Desktop/guitar data/dataset1/train/dry"
input_paths_wet_train = "/Users/michael/Desktop/guitar data/dataset1/train/wet"
input_paths_clean_valid = "/Users/michael/Desktop/guitar data/dataset1/validation/dry"
input_paths_wet_valid = "/Users/michael/Desktop/guitar data/dataset1/validation/wet"
sr = 44100
n_samples = 2 * sr
end_buffer_n_samples = 0
n_retries = 10
silence_threshold_energy = 1e-6
silence_fraction_allowed = 1
num_examples_per_epoch = 100

dataset_train = RandomAudioChunkDataset(input_paths_clean_train, input_paths_wet_train, n_samples, sr, num_examples_per_epoch=num_examples_per_epoch, silence_fraction_allowed=silence_fraction_allowed, silence_threshold_energy=silence_threshold_energy, n_retries=n_retries, end_buffer_n_samples=end_buffer_n_samples)
dataset_valid = RandomAudioChunkDataset(input_paths_clean_valid, input_paths_wet_valid, n_samples, sr, num_examples_per_epoch=num_examples_per_epoch, silence_fraction_allowed=silence_fraction_allowed, silence_threshold_energy=silence_threshold_energy, n_retries=n_retries, end_buffer_n_samples=end_buffer_n_samples)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1)
dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1)

# test dataloader
# num_silent_chunks = 0
# num_valid_chunks = 0

# per_epoch_check_dry = []
# per_epoch_check_wet = []
# colab code

seed = 12345

for epoch in range(20):

    seed += epoch # update seed for each epoch
    # TODO synchronize indexes here, valid indexes don't need to change
    random_indexes_train = dataset_train.get_permuted_indexes(seed)
    dataset_train.set_random_indexes(random_indexes_train)

    # check if random indexes change across epochs
    print("random indexes for training data: ", random_indexes_train[0])

    # check if validation data random indexes stay the same across epochs
    # print("random indexes for validation: ", dataset_valid.random_indexes)

    # input_paths_check_dry = []
    # start_idx_check_dry = []
    # input_paths_check_wet = []
    # start_idx_check_wet = []
    # # iterate over the dataset
    # print("Epoch:", epoch + 1)

    num_silent_chunks_train = 0
    num_valid_chunks_train = 0
    num_silent_chunks_valid = 0
    num_valid_chunks_valid = 0

    for batch, pbar in tqdm(enumerate(zip(dataloader_train, dataloader_valid))):
        train_data, valid_data = pbar
        
        (dry_train_x, dry_train_file_path, dry_train_start_idx), (wet_train_x, wet_train_file_path, wet_train_start_idx) = train_data
        (dry_valid_x, dry_valid_file_path, dry_valid_start_idx), (wet_valid_x, wet_valid_file_path, wet_valid_start_idx) = valid_data

        # check train data dry and wet
        assert os.path.basename(dry_train_file_path[0]) == os.path.basename(wet_train_file_path[0]), "dry and wet file path are not the same"
        assert dry_train_start_idx == wet_train_start_idx, "dry and wet start index are not the same"
        assert dry_train_x.shape == wet_train_x.shape, "dry and wet train data shape are not the same"
        
        # check valid data dry and wet
        assert os.path.basename(dry_valid_file_path[0]) == os.path.basename(wet_valid_file_path[0]), "dry and wet file path are not the same"
        assert dry_valid_start_idx == wet_valid_start_idx, "dry and wet start index are not the same"
        assert dry_valid_x.shape == wet_valid_x.shape, "dry and wet valid data shape are not the same"


        num_silent_chunks_train += dataset_train.num_silent_chunks
        num_valid_chunks_train += dataset_train.num_valid_chunks
        num_silent_chunks_valid += dataset_valid.num_silent_chunks
        num_valid_chunks_valid += dataset_valid.num_valid_chunks


    # print(num_silent_chunks_train)
    # print(num_valid_chunks_train)
    # print(num_silent_chunks_valid)
    # print(num_valid_chunks_valid)

        # print(dry_train_x.shape)
        # print(wet_train_x.shape)
        # print(dry_valid_x.shape)
        # print(wet_valid_x.shape)
        
        # input_paths_check_dry.append(dry_file_path)
        # start_idx_check_dry.append(dry_start_idx)
        # input_paths_check_wet.append(wet_file_path)
        # start_idx_check_wet.append(wet_start_idx)

    # per_epoch_check_dry.append(input_paths_check_dry)
    # per_epoch_check_dry.append(start_idx_check_dry)
    # per_epoch_check_wet.append(input_paths_check_wet)
    # per_epoch_check_wet.append(start_idx_check_wet)

# print("per_epoch_check_dry", per_epoch_check_dry)
# print("per_epoch_check_wet", per_epoch_check_wet)



