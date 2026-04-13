import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os


from dataset import PianoDataset
from model import Model

# conditional check to ensure workers in dataloader works properly to speed things up.
if __name__ == "__main__":
    dataset = PianoDataset()

    #90 percent is for training, 10 is to test the model on songs it hasnt seen yet.
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42) #select which pieces of data are training using fixed random.

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn_frame = nn.BCEWithLogitsLoss()

    #weighted loss, to prioritize aactually predicting onsets over just keeping silent / super low values.
    pos_weight = torch.ones(88) * 15.0
    loss_fn_onset = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #This is the training for the frame-only model (only returns 1 layer and stuff)
    # for epoch in range(5):
    #     total_loss = 0

    #     for mel, pr, on in loader:
    #         pred = model(mel)

    #         loss = loss_fn(pred, pr)

    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         total_loss += loss.item()

    #     print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader)}")

    #         # break  # just test one batch first

    for epoch in range(8):
        total_loss = 0

        for mel, pr, on in loader:
            frame_pred, onset_pred = model(mel) #get the frame and onset layers from the model

            loss_frame = loss_fn_frame(frame_pred, pr) #get loss for both of them
            loss_onset = loss_fn_onset(onset_pred, on)

            loss = loss_frame + loss_onset

            loss.backward() #updates weights, called backprop. 
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader)}")

    # my vscode run script in mystery spot so this saves location relative to actual file location in folder.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, "trained_models", "dual_head.pth")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path) #save the model.
