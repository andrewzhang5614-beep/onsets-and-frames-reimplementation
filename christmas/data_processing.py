import os
import librosa
import librosa.display
import matplotlib.pyplot as plot
import pretty_midi
import numpy as np

# constants / parameters
HOP_LENGTH = 600

#temp absolute path for testing.
base = os.path.dirname(__file__)
midi_path = os.path.join(base, "..", "data", "maestro_subset", "midi_files", "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")
path = os.path.join(base, "..", "data", "maestro_subset", "wav_files", "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav")


#takes in a piano roll and returns a matrix labeling only the onsets.
def get_onsets(go_piano_roll):
    onset = np.zeros_like(go_piano_roll)
    onset[0] = go_piano_roll[0]  # handle first timestep
    onset[1:] = (go_piano_roll[1:] == 1) & (go_piano_roll[:-1] == 0)
    return onset.astype(np.float32)

# process a wav file and it's respective MIDI file. 
# Returns the spectrogram, piano roll, and onset label for the files. Need to do this part.
def process_file(audio_path, midi_path):
    y, sr = librosa.load(audio_path, sr = None)
    # print(sr)

    mel = librosa.feature.melspectrogram(
        y=y, #waveform
        sr=sr, #sample rate
        n_fft=2048, # how many samples to look at in a window to determine frequency
        hop_length=HOP_LENGTH, #how much ye move the window above.
        n_mels=128  #number of frequency buckets you have. like frequencies 1 - 10 go in bucket 1, 11 - 20 b2, etc.
    )

    mel_db = librosa.power_to_db(mel)


    ###---- Processing MIDI files -----

    # get and initialize midi var, calculate matching frame rate, generate actual piano roll.
    midi = pretty_midi.PrettyMIDI(midi_path)
    frame_rate = sr / HOP_LENGTH
    piano_roll = midi.get_piano_roll(fs=frame_rate)

    # Transpose to match spectrogram (time, pitch)
    piano_roll = piano_roll.T

    # Keep piano range (21–108)
    piano_roll = piano_roll[:, 21:109]

    # Binarize key press (on/off)
    piano_roll = (piano_roll > 0).astype(np.float32)

    ###---------Meow PMf--------
    ###---------Meow PMf--------
    ###---------Meow PMf--------


    mel_db = mel_db.T # transpose so time is on y axis

    ###---------Trim both matrices to account for potential rounding issues at the end--------

    min_len = min(mel_db.shape[0], piano_roll.shape[0])

    mel_db = mel_db[:min_len]
    piano_roll = piano_roll[:min_len]

    ###-----------Meow-------------
    ###-----------Meow-------------
    ###-----------Meow-------------

    #Get onsets from the piano roll.
    onsets = get_onsets(piano_roll)

    return mel_db, piano_roll, onsets



# -------------------------------------TESTING STUFF WHILE WORKING----------------------------------------

# used to display a test spectrogram
# librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH)
# plot.colorbar()
# plot.title("Mel Spectrogram")
# plot.show()

# --------------------------------------------------------------------------------------------------------
#Used to display a test piano roll.
# print(piano_roll.shape)
# print(sr / HOP_LENGTH)  # fs

# plot.imshow(piano_roll[:2000].T, aspect='auto', origin='lower')
# plot.title("Piano Roll (first 200 frames)")
# plot.show()

# call process file on the current absolute path.
mel, pr, on = process_file(path, midi_path)
print(mel.shape)  # should be (time, 128)
print(pr.shape)   # should be (time, 88)
print(on.shape)   # should be (time, 88)