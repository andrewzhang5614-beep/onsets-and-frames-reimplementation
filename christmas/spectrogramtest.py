import os
import librosa
import librosa.display
import matplotlib.pyplot as plot
import pretty_midi

base = os.path.dirname(__file__)
# midi_path = os.path.join(base, "..", "data", "maestro_subset", "midi_files", "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")
path = os.path.join(base, "..", "data", "maestro_subset", "wav_files", "MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav")

# midi = pretty_midi.PrettyMIDI(midi_path)

y, sr = librosa.load(path, sr = None)
print(sr)

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=2048,
    hop_length=600,
    n_mels=128
)

mel_db = librosa.power_to_db(mel)

librosa.display.specshow(mel_db, sr=sr, hop_length=512)
plot.colorbar()
plot.title("Mel Spectrogram")
plot.show()