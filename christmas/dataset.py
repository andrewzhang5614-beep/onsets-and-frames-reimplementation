from data_processing import process_file
import torch
from torch.utils.data import Dataset
import os
import random

# set up cache, cache is created at the same level as this file.
base = os.path.dirname(__file__)
cache_dir = os.path.join(base, "cache")
os.makedirs(cache_dir, exist_ok=True)

class PianoDataset(Dataset):
    #init and store list of pairs of audio and midi paths.
    def __init__(self, max_len=1000):
        self.file_pairs = []
        self.max_len = max_len

        #generate paths to wav and midi files.
        base = os.path.dirname(__file__)
        audio_dir = os.path.join(base, "..", "data", "maestro_subset", "wav_files")
        midi_dir  = os.path.join(base, "..", "data", "maestro_subset", "midi_files")


        # builds up the file pairs 
        audio_files = os.listdir(audio_dir)

        for audio_file in audio_files:

            #extra safety check. 
            if audio_file.endswith(".wav"):
                midi_file = audio_file.replace(".wav", ".midi")

                # build the full path from starting from this file as a reference point 
                audio_path = os.path.join(audio_dir, audio_file)
                midi_path  = os.path.join(midi_dir, midi_file)

                # appends a tuple into the file pairs. 
                if os.path.exists(midi_path):
                    self.file_pairs.append((audio_path, midi_path))


    #return the relevant form of data for a piece of data at the given index.
    def __getitem__(self, idx):

        cache_path = os.path.join(cache_dir, f"{idx}.pt")

        # if there is a cache, check if the pre-processed data has already been stored there.
        if os.path.exists(cache_path):
            mel, pr, on = torch.load(cache_path, weights_only=False)
            # print("loading from cache")

        #else just process the data like normal and save it into the cache.
        else:
            audio_path, midi_path = self.file_pairs[idx]
            mel, pr, on = process_file(audio_path, midi_path)

            torch.save((mel, pr, on), cache_path)


        #randomizes where the sample of the song is taken from.
        if mel.shape[0] <= self.max_len:
            start = 0
        else:
            max_start = mel.shape[0] - self.max_len
            start = random.randint(0, max_start)

        end = start + self.max_len

        # --- cut length ---
        mel = mel[start:end]
        pr  = pr[start:end]
        on  = on[start:end]

        # --- Convert to tensors ---
        mel = torch.tensor(mel, dtype=torch.float32)
        pr  = torch.tensor(pr, dtype=torch.float32)
        on  = torch.tensor(on, dtype=torch.float32)

        return mel, pr, on

    #the evil one resides here...
    def __len__(self):
        return len(self.file_pairs)