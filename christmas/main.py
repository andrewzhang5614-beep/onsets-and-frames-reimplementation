import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PianoDataset
from model import Model

# conditional check to ensure workers in dataloader works properly to speed things up.
if __name__ == "__main__":
    dataset = PianoDataset()

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    #This is the training for the frame-only model (only returns 1 layer and stuff)

    # for mel, pr, on in loader:
    #     pred = model(mel)

    #     loss = loss_fn(pred, pr)

    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()

    #     print(loss.item())
    #     # break  # just test one batch first

    for mel, pr, on in loader:
        frame_pred, onset_pred = model(mel) #get the frame and onset layers from the model

        loss_frame = loss_fn(frame_pred, pr) #get loss for both of them
        loss_onset = loss_fn(onset_pred, on)

        loss = loss_frame + loss_onset

        loss.backward() #updates weights, called backprop. 
        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())
            # break  # just test one batch first