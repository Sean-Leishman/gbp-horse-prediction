"""Set-transformer over the runners in a race: self-attention across runners,
a score per runner, softmax within the race. This is the transformer variant
with published evidence behind it (CUHK LYU2102: attention across runners +
rating features beat an MLP and the favourite baseline on HK data) — NOT a
sequence model over one horse's history. No positional encoding: a race is a
set, and draw/form are already features.

Same objective and benchmark as logit_baseline.py, so the numbers compare
directly: test log-loss per race vs the market's odds-implied log-loss.
"""
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from rnn import DATA_FILE, META_COLS

BATCH_SIZE = 64


class RaceDataset(Dataset):
    """One item = (runner features [n_runners, F], winner index, odds)."""

    def __init__(self, data_file, test=False):
        df = pd.read_csv(data_file, index_col=[0])
        df = df[df.is_test == test].sort_values('date_race_id')
        self.races = []
        for _, g in df.groupby('date_race_id', sort=True):
            won = g['won'].values
            if won.sum() != 1:  # voided races / dead heats: skip
                continue
            feats = g.drop(columns=META_COLS).values.astype('float32')
            self.races.append((torch.from_numpy(feats), int(won.argmax()),
                               torch.tensor(g['odds'].values, dtype=torch.float32)))
        self.n_features = self.races[0][0].shape[1]

    def __len__(self):
        return len(self.races)

    def __getitem__(self, i):
        return self.races[i]


def collate_fn(batch):
    feats, winners, _ = zip(*batch)
    lengths = torch.tensor([len(f) for f in feats])
    padded = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    pad_mask = torch.arange(padded.shape[1])[None, :] >= lengths[:, None]
    return padded, pad_mask, torch.tensor(winners)


class RaceTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, nlayers=2):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x, pad_mask):
        h = self.encoder(self.embed(x), src_key_padding_mask=pad_mask)
        return self.score(h).squeeze(-1).masked_fill(pad_mask, float('-inf'))


def evaluate(model, loader):
    """(mean per-race log-loss, winner-pick accuracy) on a loader."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, mask, winner in loader:
            scores = model(x, mask)
            loss_sum += nn.functional.cross_entropy(
                scores, winner, reduction='sum').item()
            correct += (scores.argmax(1) == winner).sum().item()
            n += len(winner)
    return loss_sum / n, correct / n


def market_log_loss(dataset):
    """Log-loss of odds-implied probabilities on the same races."""
    total = 0.0
    for _, winner, odds in dataset.races:
        p = 1.0 / odds.clamp(min=1.01)
        total += -(p[winner] / p.sum()).log().item()
    return total / len(dataset.races)


def train_model(model, train_loader, test_loader, n_epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        total, n = 0.0, 0
        for x, mask, winner in train_loader:
            loss = nn.functional.cross_entropy(model(x, mask), winner)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(winner)
            n += len(winner)
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"epoch {epoch}: train {total / n:.4f}, "
              f"test log-loss {test_loss:.4f}, winner-pick acc {test_acc:.3f}")


if __name__ == "__main__":
    train_set = RaceDataset(DATA_FILE, test=False)
    test_set = RaceDataset(DATA_FILE, test=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             collate_fn=collate_fn)

    model = RaceTransformer(train_set.n_features)
    train_model(model, train_loader, test_loader)
    print(f"market (odds-implied) test log-loss: {market_log_loss(test_set):.4f}")
