import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 128
MAX_HISTORY = 10
DATA_FILE = "data/preprocessing/6-model-data.csv"
META_COLS = ['won', 'odds', 'morning_wap', 'date_race_id', 'offset_horse_id',
             'num_previous_races', 'is_test']


class HorseHistoryDataset(Dataset):
    """One item = (previous races incl. their outcome, current race pre-race
    features, won). Rows must be sorted (offset_horse_id, date_race_id) so a
    horse's history is the contiguous slice directly above its own row.

    Both splits load the full file: a test row may legitimately use its
    horse's pre-cutoff races as history — only the rows *predicted on* differ.
    """

    def __init__(self, data_file, test=False):
        df = pd.read_csv(data_file, index_col=[0])
        assert (df.groupby('offset_horse_id').cumcount().values
                == df['num_previous_races'].values).all(), \
            "rows not sorted (horse, date) — history slicing would cross horses"

        self.won = torch.tensor(df['won'].values, dtype=torch.float32)
        self.num_previous_races = df['num_previous_races'].values
        self.features = torch.tensor(
            df.drop(columns=META_COLS).values, dtype=torch.float32)
        self.rows = np.flatnonzero(df['is_test'].values == test)
        # kept for race-level scoring: the per-runner BCE this model trains on
        # is not comparable to logit_baseline/transformer, which score one
        # multinomial log-loss per race. See race_evaluate().
        self.race_id = torch.tensor(
            pd.factorize(df['date_race_id'].values[self.rows])[0])
        self.odds = torch.tensor(
            df['odds'].values[self.rows], dtype=torch.float32)
        self.won_rows = self.won[self.rows]

    @property
    def n_features(self):
        return self.features.shape[1]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        idx = self.rows[i]
        n = min(self.num_previous_races[idx], MAX_HISTORY)
        if n == 0:
            history = torch.zeros(1, self.n_features + 1)
        else:
            history = torch.cat(
                [self.features[idx - n:idx], self.won[idx - n:idx, None]], dim=1)
        return history, self.features[idx], self.won[idx]


def collate_fn(batch):
    histories, currents, labels = zip(*batch)
    lengths = torch.tensor([len(h) for h in histories])
    padded = nn.utils.rnn.pad_sequence(histories, batch_first=True)
    return padded, lengths, torch.stack(currents), torch.stack(labels)


class RNN(nn.Module):
    """LSTM over the horse's past races; final hidden state is concatenated
    with the current race's pre-race features to predict a win logit."""

    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features + 1, hidden_size=32,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(32 + n_features, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, history, lengths, current):
        packed = nn.utils.rnn.pack_padded_sequence(
            history, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(torch.cat([h_n[-1], current], dim=1)).squeeze(1)


def evaluate(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for history, lengths, current, y in loader:
            logits = model(history, lengths, current)
            total += loss_fn(logits, y).item() * len(y)
            n += len(y)
    return total / n


def race_evaluate(model, loader):
    """Per-race multinomial log-loss — the metric logit_baseline.py and
    transformer.py report, so the three models compare directly.

    Also the only honest metric for this model: it trains with pos_weight to
    counter the ~1-in-10 class imbalance, which shifts every logit up by
    roughly a constant. That wrecks unweighted per-runner BCE (it scores worse
    than predicting the base rate) but cancels exactly in a within-race
    softmax, which is invariant to a uniform shift.

    Requires an unshuffled loader: logits are matched to rows by order.
    """
    from logit_baseline import race_log_loss  # deferred: logit_baseline imports us
    model.eval()
    scores = []
    with torch.no_grad():
        for history, lengths, current, _ in loader:
            scores.append(model(history, lengths, current))
    ds = loader.dataset
    return race_log_loss(torch.cat(scores), ds.race_id, ds.won_rows).item()


def market_race_log_loss(dataset):
    """Odds-implied log-loss on the same races, normalised within each race."""
    from logit_baseline import race_log_loss
    p = 1.0 / dataset.odds.clamp(min=1.01)
    n = int(dataset.race_id.max()) + 1
    total = torch.zeros(n).scatter_add(0, dataset.race_id, p)
    return race_log_loss((p / total[dataset.race_id]).log(),
                         dataset.race_id, dataset.won_rows).item()


def train_model(model, train_loader, test_loader, n_epochs=10):
    train_set = train_loader.dataset
    pos = train_set.won[train_set.rows].sum()
    pos_weight = (len(train_set) - pos) / pos  # ~1 winner per ~10 runners
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_loss_fn = nn.BCEWithLogitsLoss()  # unweighted log-loss for reporting
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total, seen = 0.0, 0
        for history, lengths, current, y in train_loader:
            logits = model(history, lengths, current)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            seen += len(y)
        print(f"epoch {epoch}: train loss {total / seen:.4f}, "
              f"test per-runner BCE {evaluate(model, test_loader, eval_loss_fn):.4f}, "
              f"test RACE log-loss {race_evaluate(model, test_loader):.4f}")


if __name__ == "__main__":
    train_set = HorseHistoryDataset(DATA_FILE, test=False)
    test_set = HorseHistoryDataset(DATA_FILE, test=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             collate_fn=collate_fn)

    model = RNN(train_set.n_features)
    train_model(model, train_loader, test_loader)
    print(f"market (odds-implied) race log-loss: "
          f"{market_race_log_loss(test_set):.4f}")
