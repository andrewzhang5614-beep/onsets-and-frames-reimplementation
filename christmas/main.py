import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PianoDataset
from model import Model


dataset = PianoDataset()

# print(len(dataset))  # should be ~13 or whatever you added

# mel, pr, on = dataset[0]

# print(mel.shape)
# print(pr.shape)
# print(on.shape)

loader = DataLoader(dataset, batch_size=4, shuffle=True)


# for mel, pr, on in loader:
#     print(mel.shape)
#     print(pr.shape)
#     print(on.shape)
#     break







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