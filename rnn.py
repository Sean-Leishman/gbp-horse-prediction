import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torch import nn

from memory_profiler import profile

from timeit import default_timer as timer

BATCH_SIZE = 128

class HorseHistoryDataset(Dataset):
    def __init__(self, data_file, transform=None) -> None:
        df = pd.read_csv(data_file, index_col=[0])
        
        self.num_previous_races = torch.tensor(df['num_previous_races'].values)
        df = df.drop(['num_previous_races', 'offset_horse_id'], axis=1)

        self.values = torch.tensor(df.values)
        
        self.transform = transform

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx):
        num_races = self.num_previous_races[idx].item()
        if num_races == 0:
            x_data = self.values[idx:idx+1]
            y_data = self.values[idx]
        else:
            x_data = self.values[idx - num_races:idx]
            y_data = self.values[idx]

        return x_data, y_data


class RNN(nn.Module):
    def __init__(self) -> None:
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=35, hidden_size=12,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=12, out_features=35)

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

    print(f"Length of train loader: {len(train_loader)}, Number of batches: {len(train_loader) / BATCH_SIZE}")

    sum_t = 0
    for epoch in range(n_epochs):
        idx = 0
        num_batches = len(train_loader)
        total_loss = 0

        start = timer()
        for X_batch, y_batch in train_loader:
            start_t = timer()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            idx += 1

            end_t = timer()
            sum_t += (end_t - start_t)

            if (idx % 1000 == 0):
                end = timer()
                
                print(f"Batch Number: {idx}, Time: {end-start}, TrainTime: {sum_t}")
                start = timer()
                sum_t = 0

        avg_loss = total_loss / num_batches
        print(f"Train loss: {avg_loss}, Batches: {idx}")

        model.eval()
        num_batches = len(train_loader)
        total_loss = 0

        model.eval()
        with torch.no_grad():
            pass
            # for X, y in test_loader:
            #     output = model(X)
            #     total_loss += loss_fn(output, y).item()

        avg_loss = total_loss / num_batches
        print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    output = r([])
    model.eval()
    model(data_loader)


def collate_fn(batch):
    batch = [(b[0].tolist(), b[1]) for b in batch]
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    sequences_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in sequences], batch_first=True, padding_value=0)
    return sequences_padded, torch.stack(labels)


if __name__ == "__main__":
    print("LOADING DATA ...")
    dataset = HorseHistoryDataset("data/no_cat_full_features.csv")
    print("DATASET LOADED")
    train_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                              persistent_workers=True,
                              num_workers=4)
    print("LOADED DATA")
    # test_loader = DataLoader(HorseHistoryDataset("data/raw/processed_normalized_full_data_with_features","data/raw/processed_normalized_full_data_with_horse_history", "data/raw/fff.csv", filtered_data=fd_test_data), shuffle=True, batch_size=32)
    
    print("LOADING RNN ...")
    model = RNN()

    print("TRAINING MODEL ...")
    train_model(train_loader, model)

    # X_test, y_test = test_dataset
