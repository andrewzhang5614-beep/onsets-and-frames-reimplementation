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

    for mel, pr, on in loader:
        pred = model(mel)

        loss = loss_fn(pred, pr)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(loss.item())
        # break  # just test one batch first