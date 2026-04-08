from dataset import PianoDataset
from torch.utils.data import DataLoader


dataset = PianoDataset()

# print(len(dataset))  # should be ~13 or whatever you added

# mel, pr, on = dataset[0]

# print(mel.shape)
# print(pr.shape)
# print(on.shape)

loader = DataLoader(dataset, batch_size=4, shuffle=True)


for mel, pr, on in loader:
    print(mel.shape)
    print(pr.shape)
    print(on.shape)
    break