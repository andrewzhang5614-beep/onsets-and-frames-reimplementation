from data_processing import process_file
import torch
from torch.utils.data import Dataset
import numpy as np  # optional (only if needed)
import os


class PianoDataset:
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
        audio_path, midi_path = self.file_pairs[idx]
        mel, pr, on = process_file(audio_path, midi_path)


        # --- Handle length (crop) ---
        mel = mel[:self.max_len]
        pr  = pr[:self.max_len]
        on  = on[:self.max_len]

        # --- Convert to tensors ---
        mel = torch.tensor(mel, dtype=torch.float32)
        pr  = torch.tensor(pr, dtype=torch.float32)
        on  = torch.tensor(on, dtype=torch.float32)

        return mel, pr, on

    #the evil one resides here...
    def __len__(self):
        return len(self.file_pairs)