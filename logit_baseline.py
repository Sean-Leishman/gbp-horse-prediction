"""Benter-style conditional logit: a linear score per runner, softmax over the
runners in each race, trained to pick the winner (Bolton & Chapman 1986,
Benter 1994). This is the canonical baseline any fancier model has to beat.
Also reports the market's own log-loss from odds-implied probabilities.
"""
import pandas as pd
import torch
from torch import nn

from rnn import DATA_FILE, META_COLS


def load_split(test):
    df = pd.read_csv(DATA_FILE, index_col=[0])
    df = df[df.is_test == test]
    X = torch.tensor(df.drop(columns=META_COLS).values, dtype=torch.float32)
    race = torch.tensor(pd.factorize(df['date_race_id'])[0])
    won = torch.tensor(df['won'].values, dtype=torch.float32)
    odds = torch.tensor(df['odds'].values, dtype=torch.float32)
    return X, race, won, odds


def race_log_loss(scores, race, won):
    """Mean over races of -log softmax(score)[winner]."""
    n_races = int(race.max()) + 1
    m = torch.full((n_races,), -torch.inf).scatter_reduce(
        0, race, scores, reduce='amax')
    sumexp = torch.zeros(n_races).scatter_add(0, race, (scores - m[race]).exp())
    log_p = scores - (m + sumexp.log())[race]
    winners = torch.zeros(n_races).scatter_add(0, race, won)
    return -(log_p * won).sum() / winners.clamp(min=1).sum()


def market_log_loss(odds, race, won):
    """Log-loss of odds-implied probabilities, normalised within each race."""
    p = 1.0 / odds.clamp(min=1.01)  # odds <= 1 are missing/bad rows
    n_races = int(race.max()) + 1
    total = torch.zeros(n_races).scatter_add(0, race, p)
    return race_log_loss((p / total[race]).log(), race, won)


if __name__ == "__main__":
    X_train, race_train, won_train, _ = load_split(test=False)
    X_test, race_test, won_test, odds_test = load_split(test=True)

    model = nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for epoch in range(200):
        loss = race_log_loss(model(X_train).squeeze(1), race_train, won_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            with torch.no_grad():
                test_loss = race_log_loss(
                    model(X_test).squeeze(1), race_test, won_test)
            print(f"epoch {epoch}: train {loss.item():.4f}, test {test_loss.item():.4f}")

    print(f"market (odds-implied) test log-loss: "
          f"{market_log_loss(odds_test, race_test, won_test).item():.4f}")
