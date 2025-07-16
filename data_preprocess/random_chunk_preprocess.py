import torchaudio
import torch as tr
import pyloudnorm as pyln
from torch import Tensor as T
from torch.utils.data import Dataset
import logging
import os
from collections import defaultdict
from typing import Dict, Optional, List, Any, Tuple, Type, Union
from tqdm import tqdm



class RandomAudioChunkDataset(Dataset):
    
    """
    silence_fraction_allowed: minimum silence fraction of audio allowed, discard otherwise
    silence_threshold_energy: used to detect silence fraction
    n_retries: number of tries to generate random index for a specific audio file
    check_dataset: whether to perform check on whether files are "suitable" for a set of conditions
    min_suitable_files_fraction: minimum fraction of files allowed, discard otherwise
    end_buffer_n_samples: 
    should_peak_norm: 
    peak_norm_db: peak norm
    """
    
    def __init__(
            self,
            input_dir_dry: str,
            input_dir_wet: str,
            n_samples: int,
            num_examples_per_epoch: int = 10000,
            sr: float = 44100,
            ext: str = "wav",
            silence_fraction_allowed: float = 0.2,
            silence_threshold_energy: float = 1e-6,  # Around -60 dBFS
            n_retries: int = 10,
            check_dataset: bool = True,
            min_suitable_files_fraction: int = 0.5,
            end_buffer_n_samples: int = 0,
            should_peak_norm: bool = False,
            peak_norm_db: float = -1.0,
            seed: int = 12345,
    ) -> None:
        super().__init__()

        self.input_dir_dry = input_dir_dry
        self.input_dir_wet = input_dir_wet
        self.n_samples = n_samples
        self.num_examples_per_epoch = num_examples_per_epoch
        self.random_indexes = None
        self.sr = sr
        self.ext = ext
        self.silence_fraction_allowed = silence_fraction_allowed
        self.silence_threshold_energy = silence_threshold_energy
        self.n_retries = n_retries
        self.check_dataset = check_dataset
        self.min_suitable_files_fraction = min_suitable_files_fraction
        self.end_buffer_n_samples = end_buffer_n_samples
        self.should_peak_norm = should_peak_norm
        self.peak_norm_db = peak_norm_db
        self.max_n_consecutive_silent_samples = int(silence_fraction_allowed * n_samples)
        self.num_silent_chunks = 0
        self.num_valid_chunks = 0
        self.seed = seed

        # get a list of paths
        input_paths_dry, file_names_check = self.get_file_paths(input_dir_dry, ext)
        input_paths_wet, file_names_check = self.get_file_paths(input_dir_wet, ext)
        self.file_names_check = file_names_check
    
        # print('input_paths')
        # print(input_paths_dry)    
        # print(input_paths_wet)    
        # print('file_names_check')
        # print(file_names_check)

        total_n_samples = 0
        filtered_input_paths_dry = []
        filtered_input_paths_wet = []
        
        for input_path_dry, input_path_wet in zip(input_paths_dry, input_paths_wet):
            file_info_dry = torchaudio.info(input_path_dry)
            file_info_wet = torchaudio.info(input_path_wet)
            
            # 這邊都是single file邏輯
            if file_info_dry.num_frames < n_samples:
                print(f"Too short, removing: {input_path_dry}")
                continue

            if file_info_wet.num_frames < n_samples:
                print(f"Too short, removing: {input_path_wet}")
#                 log.debug(f"Too short, removing: {input_path}")
                continue
        
            if file_info_dry.sample_rate != sr:
                print(f"Bad sample rate of {file_info_dry.sample_rate}, removing: {input_path_dry}")
                continue

            if file_info_wet.sample_rate != sr:
                print(f"Bad sample rate of {file_info_wet.sample_rate}, removing: {input_path_wet}")
                continue

            total_n_samples += file_info_dry.num_frames
            filtered_input_paths_dry.append(input_path_dry)
            filtered_input_paths_wet.append(input_path_wet)
            

        print(f"Filtered down to {len(filtered_input_paths_dry)} input files")
#         log.info(f"Filtered down to {len(filtered_input_paths)} input files")
        print(f"Found {total_n_samples / sr:.0f} seconds ({total_n_samples / sr / 60.0:.2f} minutes) of audio")
#         log.info(f"Found {total_n_samples / sr:.0f} seconds ({total_n_samples / sr / 60.0:.2f} minutes) of audio")
        
        assert len(filtered_input_paths_dry) > 0
        
        self.input_paths_dry = filtered_input_paths_dry
        self.input_paths_wet = filtered_input_paths_wet


        assert len(filtered_input_paths_dry) == len(filtered_input_paths_wet), "input_paths_dry and input_paths_wet are not the same"     
        
        # print('self.input_paths')
        # print(self.input_paths_dry)
        # print(self.input_paths_wet)
        
        # TODO 這邊要重寫
#         if check_dataset:
#             assert self.check_dataset_for_suitable_files(n_samples,
#                                                          min_suitable_files_fraction
#                                                          end_buffer_n_samples), \
#                 "Could not find a suitable non-silent audio chunk in the dataset"

        # set random indexes upon initialization
        self.random_indexes = self.get_permuted_indexes(self.seed)

    # NOTE suitable files 是符合user定義 sample length 的 audio (這邊先check dry file就好，因為wet file跟dry file照理來說是要一樣的)
    def check_dataset_for_suitable_files(self,
                                         input_paths: list,
                                         n_samples: int,
                                         min_suitable_files_fraction: float,
                                         end_buffer_n_samples: int = 0) -> bool:
        
        min_n_suitable_files = int(min_suitable_files_fraction * len(input_paths))
        min_n_suitable_files = max(1, min_n_suitable_files)
        n_suitable_files = 0

        # NOTE 在for loop裡面 random sample 去找silent段
        for file_path in tqdm(input_paths):
            # retries 應該是load同一個file但拿不同random index
            for _ in range(self.n_retries):
                audio_chunk = self.find_audio_chunk_in_file(file_path, n_samples, index, end_buffer_n_samples)
                if audio_chunk is not None:
                    n_suitable_files += 1
                    break

        print(f"Found {n_suitable_files} suitable files out of {len(input_paths)} files ")
        print(f"{n_suitable_files / len(input_paths) * 100:.2f}%")
        
        print(f"Found {n_suitable_files} suitable files out of {len(input_paths)} files "
                 f"({n_suitable_files / len(input_paths) * 100:.2f}%)")
#         log.info(f"Found {n_suitable_files} suitable files out of {len(self.input_paths)} files "
#                  f"({n_suitable_files / len(self.input_paths) * 100:.2f}%)")
        
        
        return n_suitable_files >= min_n_suitable_files

    # NOTE 用energy找silence段
    def check_for_silence(self, audio_chunk: T) -> bool:
        window_size = self.max_n_consecutive_silent_samples
        hop_len = window_size // 4
        energy = audio_chunk ** 2
        # print(audio_chunk.shape)
        
        unfolded = energy.unfold(dimension=-1, size=window_size, step=hop_len)
        mean_energies = tr.mean(unfolded, dim=-1, keepdim=False)
        n_silent = (mean_energies < self.silence_threshold_energy).sum().item()
        
        return n_silent > 0

    # per file
    def find_audio_chunk_in_file(self,
                                 file_path: str,
                                 n_samples: int,
                                 index: int,
                                 end_buffer_n_samples: int = 0) -> Optional[Tuple[T, int]]:
        
        
        # print('debug---')
        # print("index", index)
        file_n_samples = torchaudio.info(file_path).num_frames

        # NOTE 當 user define > 實際audio len
        
#         print("end_buffer_n_samples", end_buffer_n_samples)
#         print("file_n_samples", file_n_samples)
#         print("n_samples", n_samples)
        
        if n_samples > file_n_samples - end_buffer_n_samples:
            print(f"n_samples {n_samples} is greater than file_n_samples {file_n_samples} - end_buffer_n_samples {end_buffer_n_samples}")
            return None
        
        
        n_samples = int(n_samples)
        
        # 猜通常start_idx會是1
#         print("file_n_samples - n_samples - end_buffer_n_samples + 1: ", file_n_samples - n_samples - end_buffer_n_samples + 1)
        
        # 這邊其實在找 end index 最多可以落在哪裡，避免隨機生random index的時候超出範圍
#         high = int(file_n_samples - n_samples - end_buffer_n_samples + 1)
#         start_idx = self.randint(0, high)
#         n_samples = int(n_samples)


        try:
            audio_chunk, sr = torchaudio.load(file_path, frame_offset=index, num_frames=n_samples,)
        except RuntimeError:
            print(f"open audio fails, starting index {index} is greater than total number of frames {torchaudio.info(file_path).num_frames}")
            audio_chunk = tr.tensor(0)
            return None

        # Convert to mono by averaging channels
        if audio_chunk.shape[0] > 1:
            audio_chunk = audio_chunk.mean(dim=0, keepdim=True)  # shape: (1, samples)
        else:
            pass  # already mono
        
        # BUG temporary workaround
        if audio_chunk.shape[-1] == 0:
            return None

        # NOTE 找silence段
        if self.check_for_silence(audio_chunk):
            print("Skipping audio chunk because of silence")

            # count number of silent chunks
            self.num_silent_chunks += 1
#             log.debug("Skipping audio chunk because of silence")
            return None
        
        else:
            self.num_valid_chunks += 1
        
        return audio_chunk, index

    # 先找到 "合適" 的 audio chunk, 然後再拿一個random idx
    def search_dataset_for_audio_chunk(self, file_path:str, n_samples: int, index:list, end_buffer_n_samples: int = 0.) -> (T, str, int, int):
        

#         file_path_pool = list(self.input_paths)
#         file_path = self.choice(file_path_pool)
#         file_path_pool.remove(file_path)
        audio_chunk = None
        n_attempts = 0
        
        start_indexes = []
        audio_chunks = []

        # breakpoint()

        for random_index in index:
            # print('random_index from search_dataset_for_audio_chunk')
            # print(random_index)
            # print('index')
            # print(index)
            
            # debug for potentially wrong index
            
            # breakpoint()
            audio_chunk = self.find_audio_chunk_in_file(file_path, n_samples, random_index, end_buffer_n_samples)
            
            if audio_chunk is None:
                n_attempts += 1
                continue
                
                # NOTE TBD
#                 if n_attempts >= self.n_retries:
#                     assert file_path_pool, "This should never happen if `check_dataset_for_suitable_files` was run"
#                     file_path = self.choice(file_path_pool)
#                     file_path_pool.remove(file_path)
#                     n_attempts = 0
            
            audio_chunk_, start_idx = audio_chunk

            # print("audio_chunk", audio_chunk)
#             if audio_chunk is None:
#                 import pdb; pdb.set_trace()

            start_indexes.append(start_idx)
            audio_chunks.append(audio_chunk_)
        
        
        try:
            audio_chunks = tr.cat(audio_chunks, dim=0)
        # BUG RuntimeError: torch.cat(): expected a non-empty list of Tensors
        except RuntimeError:
            audio_chunks = tr.tensor(0)
        
        
        # NOTE TBD
#         ch_idx = 0
#         if audio_chunk.size(0) > 1:
#             ch_idx = randint(0, audio_chunk.size(0))
#             audio_chunk = audio_chunk[ch_idx, :].view(1, -1)

#         return audio_chunk, file_path, ch_idx, start_idx

        return audio_chunks, file_path, start_indexes


    def peak_normalize(self, audio: T) -> T:
        assert audio.ndim == 2
        audio_np = audio.T.numpy()
        audio_norm_np = pyln.normalize.peak(audio_np, self.peak_norm_db) #https://github.com/csteinmetz1/pyloudnorm
        audio_norm = tr.from_numpy(audio_norm_np.T)
        
        
        return audio_norm

    def __len__(self) -> int:
        # NOTE workaround: number of iterations is specified separately
        return 1
    
    def __getitem__(self, index):
 
        # dataloader passes index in sequential order by default, so just use our mapping
        random_indexes_per_file = self.random_indexes[index]
        indexed_file_path_dry = self.input_paths_dry[index]
        indexed_file_path_wet = self.input_paths_wet[index]


        # audio_chunk shape: (num_audio_chunks, audio_chunk_length)
        audio_chunk_dry, file_path_dry, start_idx_dry = self.search_dataset_for_audio_chunk(indexed_file_path_dry, self.n_samples, random_indexes_per_file, self.end_buffer_n_samples)
        audio_chunk_wet, file_path_wet, start_idx_wet = self.search_dataset_for_audio_chunk(indexed_file_path_wet, self.n_samples, random_indexes_per_file, self.end_buffer_n_samples)

        # take shorter index list
        if start_idx_dry != start_idx_wet:
            if len(start_idx_dry) > len(start_idx_wet):
                start_idx_dry_new = []
                for index in start_idx_dry:
                    if index in start_idx_wet:
                        start_idx_dry_new.append(index)
                # update index list
                start_idx_dry = start_idx_dry_new
                audio_chunk_dry = audio_chunk_dry[:len(start_idx_dry)]
            
            elif len(start_idx_wet) > len(start_idx_dry):
                start_idx_wet_new = []
                for index in start_idx_wet:
                    if index in start_idx_dry:
                        start_idx_wet_new.append(index)
                # update index list
                start_idx_wet = start_idx_wet_new
                audio_chunk_wet = audio_chunk_wet[:len(start_idx_wet)]
            else:
                pass

        assert os.path.basename(file_path_dry) == os.path.basename(file_path_wet), "file_path_dry and file_path_wet are not the same"
        assert start_idx_dry == start_idx_wet, "start_idx_dry and start_idx_wet are not the same"


        if self.should_peak_norm:
            audio_chunk_dry = self.peak_normalize(audio_chunk_dry)
            audio_chunk_wet = self.peak_normalize(audio_chunk_wet)

        return (audio_chunk_dry, file_path_dry, start_idx_dry), (audio_chunk_wet, file_path_wet, start_idx_wet)
    
    @staticmethod
    def get_file_paths(input_dir: str, ext: str) -> List[str]:
        """
        return list of audio file paths
        """

        print(input_dir)
        assert os.path.isdir(input_dir)
        input_paths = []
        file_names_check = None
        for root_dir, _, file_names in os.walk(input_dir):
            file_names_check = file_names
            # print(root_dir, file_names)   
            for file_name in file_names:
                if file_name.endswith(ext) and not file_name.startswith("."):
                    input_paths.append(os.path.join(root_dir, file_name))

        input_paths = sorted(input_paths)
        file_names_check = sorted(file_names_check) # sort file names to make sure the order is the same as the input_paths

        print(f"Found {len(input_paths)} files in {input_dir}")
#         log.info(f"Found {len(input_paths)} files in {input_dir}")
        assert len(input_paths) > 0

        return input_paths, file_names_check
    
    def randint(self, low: int, high: int, n: int = 1) -> Union[int, T]:
        x = tr.randint(low=low, high=high, size=(n,))
        if n == 1:
            return x.item()
        return x
    
    def choice(self, items: List[Any]) -> Any:
        assert len(items) > 0
        idx = randint(0, len(items))
        return items[idx]
    
    def set_random_indexes(self, random_indexes: List[int]) -> None:
        self.random_indexes = random_indexes
    
    def get_permuted_indexes(self, seed: int) -> List[int]:
        """
        inputs:
            seed: seed for random number generator

        outputs:
            list_of_permuted_indexes: list of random indexes for each audio file

        assuming wet and dry have the same sr, and file length, we can use the same random indexes for both wet and dry
        """

        list_of_permuted_indexes = []
        
        file_list = sorted(os.listdir(self.input_dir_dry))
        for i in range(len(file_list)):
            file_list[i] = os.path.join(self.input_dir_dry, file_list[i])

        for input_path in file_list:
            if os.path.basename(input_path) == '.DS_Store':
                continue
            file_info = torchaudio.info(input_path)
            total_number_of_frames = file_info.num_frames
            random_indexes = random_permute(self.n_retries, total_number_of_frames, self.n_samples, self.end_buffer_n_samples, seed=seed)
            list_of_permuted_indexes.append(random_indexes)

        # breakpoint()

        return list_of_permuted_indexes
    
def randint(low: int, high: int, n: int = 1) -> Union[int, T]:
    x = tr.randint(low=low, high=high, size=(n,))
    if n == 1:
        return x.item()
    return x

# test random permute
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