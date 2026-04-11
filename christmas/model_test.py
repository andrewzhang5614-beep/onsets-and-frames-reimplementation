import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from dataset import PianoDataset
from model import Model


# conditional check to ensure workers in dataloader works properly to speed things up.
if __name__ == "__main__":
    dataset = PianoDataset(max_len=900)

    #90 percent is for training, 10 is to test the model on songs it hasnt seen yet.
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42) #select which pieces of data are training using fixed random.

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )


    # my vscode run script in mystery spot so this saves location relative to actual file location in folder.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "trained_models", "dual_head.pth")

    model = Model()
    model.load_state_dict(torch.load(save_path))
    model.eval()


    # ---- TEST (Option A: single sample) ----
    mel, pr, on = test_dataset[0]

    mel = mel.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        frame_pred, onset_pred = model(mel)

    # Apply sigmoid (IMPORTANT)
    frame_pred = torch.sigmoid(frame_pred)
    onset_pred = torch.sigmoid(onset_pred)

    # Threshold
    frame_bin = (frame_pred > 0.14)
    onset_bin = (onset_pred > 0.15)

    # Post-processing
    combined = frame_bin & onset_bin

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))

    # Ground truth
    axs[0].imshow(pr.T, aspect='auto', origin='lower')
    axs[0].set_title("Ground Truth")

    # Frame prediction
    axs[1].imshow(frame_bin[0].T, aspect='auto', origin='lower')
    axs[1].set_title("Frame Prediction")

    # Onset prediction
    axs[2].imshow(onset_bin[0].T, aspect='auto', origin='lower')
    axs[2].set_title("Onset Prediction")

    # Combined (post-processing)
    axs[3].imshow(combined[0].T, aspect='auto', origin='lower')
    axs[3].set_title("Frame + Onset")

    plt.tight_layout()
    plt.show()