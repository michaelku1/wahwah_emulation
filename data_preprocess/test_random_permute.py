"""
check:

- batch wise indexes are different for each batch
...

"""

import torch as tr

def set_global_seed(seed):
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.cuda.manual_seed_all(seed)
    tr.backends.cudnn.deterministic = True
    tr.backends.cudnn.benchmark = False

def randint(low: int, high: int, n: int = 1):
    x = tr.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x

def random_permute(n_retries, file_n_samples, n_samples, end_buffer_n_samples, seed=12345):
    '''
    randomly permute index for random chunking
    '''
    
    tr.manual_seed(seed)

    # assuming input audio data is >> user defined audio chunk (e.g ~2s)
    high = int(file_n_samples - n_samples - end_buffer_n_samples + 1)
    
    random_indexes = []
    for _ in range(n_retries):
        start_idx = randint(0, high)
        random_indexes.append(start_idx)

    return random_indexes


if __name__ == "__main__":
    # set_global_seed(12345)
    # random_indexes = random_permute(n_retries=10, file_n_samples=1000, n_samples=100, end_buffer_n_samples=0)
    # print(random_indexes)

    # [13, 12, 258, 186, 564, 108, 194, 529, 339, 494]

    # load a batch audio and save each audio chunk
    import soundfile as sf
    import torchaudio
    import os
    from random_chunk_preprocess import RandomAudioChunkDataset

    dry_path = "/mnt/gestalt/home/mku666/EGDB_DI/"
    wet_path = "/mnt/gestalt/home/mku666/NA_Wah/NA_WahFilter_7.5_Power_True_Bypass_False/"

    num_examples_per_epoch = 1000
    sr = 44100
    n_samples = sr * 2
    silence_fraction_allowed = 0.8
    silence_threshold_energy = 1e-6
    n_retries = 3
    end_buffer_n_samples = 0
    should_peak_norm = True

    dataset_train = RandomAudioChunkDataset(dry_path,
                                        wet_path,
                                        n_samples,
                                        num_examples_per_epoch=num_examples_per_epoch,
                                        sr=sr,
                                        silence_fraction_allowed=silence_fraction_allowed,
                                        silence_threshold_energy=silence_threshold_energy,
                                        n_retries=n_retries,
                                        end_buffer_n_samples=end_buffer_n_samples,
                                        should_peak_norm = should_peak_norm,
                                        seed=12345)


    dry_train, wet_train = next(iter(dataset_train))

    dry_train = dry_train[0]
    wet_train = wet_train[0]

    for i in range(n_retries):
        dry_chunk = dry_train[i, :].unsqueeze(0)
        wet_chunk = wet_train[i, :].unsqueeze(0)

        torchaudio.save(f"../audio_samples_no_peak_norm/dry_chunk_{i}.wav", dry_chunk, sr, backend='soundfile')
        torchaudio.save(f"../audio_samples_no_peak_norm/wet_chunk_{i}.wav", wet_chunk, sr, backend='soundfile')

