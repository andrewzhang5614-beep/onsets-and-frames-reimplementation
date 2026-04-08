from dataset import PianoDataset



dataset = PianoDataset()

print(len(dataset))  # should be ~13 or whatever you added

mel, pr, on = dataset[0]

print(mel.shape)
print(pr.shape)
print(on.shape)