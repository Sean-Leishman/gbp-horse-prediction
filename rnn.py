import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import nn


class HorseHistoryDataset(Dataset):
    def __init__(self, raw_data_file, filtered_data_file,  index_file, filtered_data=None, transform=None) -> None:
        self.raw_data = pd.read_csv(raw_data_file, index_col=[0])
        if filtered_data_file == "":
            self.filtered_data = filtered_data
        else:
            self.filtered_data = pd.read_csv(filtered_data_file, index_col=[0])
        self.horse_history_index = pd.read_csv(index_file, index_col=[0])

        self.raw_data = self.raw_data.set_index(["date_race_id", "horse_ids"])
        self.filtered_data = self.filtered_data.set_index(
            ["date_race_id", "horse_ids"])
        self.horse_history_index = self.horse_history_index.set_index(
            ["race_id", "horse_ids"])

        self.transform = transform

    def __len__(self) -> int:
        return len(self.filtered_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print("index here:", idx)
        horse = self.filtered_data.iloc[idx]
        if self.transform:
            pass

        try:
            hist = self.horse_history_index.loc[(
                horse.race_id, horse.name[1])]['date_race_id'].values
            indexes = [(x, horse.name[1]) for x in hist]
            x_data = torch.tensor(
                self.raw_data.loc[self.raw_data.index.isin(indexes)].values.astype('float64'))
            # Gets the horse race that is currently occuring. Will be predicting the track stats too
            # Could just extract 'won' variable
            y_data = torch.tensor(horse.values.astype('float64'))

        except KeyError as e:
            print("Key Error")
            return None

        return x_data, y_data


class RNN(nn.Module):
    def __init__(self) -> None:
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=178, hidden_size=5,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1:]
        output = self.fc1(torch.relu(output))
        return output


def train_model(train_loader, model, test_loader=None):
    n_epochs = 200
    loss_fn = torch.nn.HingeEmbeddingLoss()
    total_loss = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(n_epochs):
        num_batches = len(train_loader)
        total_loss = 0
        for X_batch, y_batch in train_loader:
            print(X_batch)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}")

        model.eval()
        num_batches = len(train_loader)
        total_loss = 0

        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                output = model(X)
                total_loss += loss_fn(output, y).item()

        avg_loss = total_loss / num_batches
        print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    model(data_loader)


def collate_fn(batch):
    batch = [(b[0].tolist(), b[1]) for b in batch]
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    sequences_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.Tensor(s) for s in sequences], batch_first=True, padding_value=0)
    return sequences_padded, torch.tensor(labels)


if __name__ == "__main__":
    filtered_data_file = "data/raw/processed_normalized_full_data_with_horse_history.csv"
    df = pd.read_csv(filtered_data_file, index_col=[0])
    fd_train_data, fd_test_data = train_test_split(df)

    # add collate function
    train_loader = DataLoader(HorseHistoryDataset("data/raw/processed_normalized_full_data_with_features.csv",
                              "", "data/raw/fff.csv", filtered_data=fd_train_data), shuffle=True, batch_size=16, collate_fn=collate_fn)
    # test_loader = DataLoader(HorseHistoryDataset("data/raw/processed_normalized_full_data_with_features","data/raw/processed_normalized_full_data_with_horse_history", "data/raw/fff.csv", filtered_data=fd_test_data), shuffle=True, batch_size=32)

    model = RNN()
    train_model(train_loader, model)

    # X_test, y_test = test_dataset
