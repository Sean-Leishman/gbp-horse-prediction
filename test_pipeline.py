"""Smallest end-to-end check: synthetic races -> feature stages -> temporal
split -> a few training steps of both models. Asserts the past-only property
(no look-ahead leakage). Run: python test_pipeline.py
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocessing import Preprocessor, FEATURE_COLS


def make_synthetic_df(n_races=60, runners=6):
    rng = np.random.default_rng(0)
    rows = []
    for r in range(n_races):
        horses = rng.choice(30, size=runners, replace=False)
        winner = rng.integers(runners)
        for i, h in enumerate(horses):
            rows.append({
                'race_id': r,
                'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=int(r)),
                'horse_ids': int(h),
                'jockey_ids': int(h % 5), 'trainer_ids': int(h % 4),
                'sire_id': int(h % 7) + 100, 'dam_id': int(h % 9) + 200,
                'dam_sire_id': int(h % 5) + 300,
                'going': int(rng.integers(0, 11)),
                'distance': int(rng.integers(1000, 4000)),
                'distance_categories': int(rng.integers(0, 10)),
                'race_class': int(rng.integers(1, 8)),
                'race_type': int(rng.integers(0, 3)),
                'race_handicap': int(rng.integers(0, 2)),
                'draws': int(i),
                'horse_ages': int(rng.integers(0, 5)),
                'horse_weight': int(rng.integers(120, 170)),
                'top_speeds': int(rng.integers(1, 100)),
                'ratings': int(rng.integers(1, 100)),
                'official_ratings': int(rng.integers(1, 100)),
                'odds': float(rng.uniform(1.5, 30)),
                'won': int(i == winner),
                'places': 1 if i == winner else int(i + 2 if i < winner else i + 1),
                'length': float(rng.random() * 10),
            })
    df = pd.DataFrame(rows)
    for t in (0, 1, 2):
        df[f'race_type__{t}'] = (df['race_type'] == t).astype(int)
    return df


def main():
    p = Preprocessor()
    p.df = make_synthetic_df().sort_values('date')
    p.df['date_race_id'] = pd.factorize(p.df['race_id'])[0]
    p.preprocess_columns()
    p.compute_horse_features(['going', 'distance'])
    p.compute_auxillary_features_group()
    p.compute_pedigree_group()

    # past-only: first appearances must have zero/default history stats
    by_date = p.df.sort_values('date')
    assert (by_date.drop_duplicates('horse_ids')['horse_win_percents'] == 0).all()
    assert (by_date.drop_duplicates('sire_id')['sire_win_percent'] == 0).all()
    assert (by_date.drop_duplicates('horse_ids')['elo_rating'] == 1500).all()
    assert p.df['elo_rating'].nunique() > 1  # ratings actually move

    # past-only: a row's sire stat equals the mean of strictly-earlier rows
    df = p.df.reset_index(drop=True)
    idx = df.index[df.sire_id == df.iloc[-1].sire_id]
    expected = df.loc[idx[:-1], 'won'].mean() if len(idx) > 1 else 0.0
    assert abs(df.loc[idx[-1], 'sire_win_percent'] - expected) < 1e-9

    p.select_columns()
    os.makedirs("data/preprocessing", exist_ok=True)
    p.train_test_split()

    assert not p.df[FEATURE_COLS].isna().any().any()
    assert p.df['is_test'].any() and (~p.df['is_test']).any()

    from rnn import HorseHistoryDataset, RNN, collate_fn, train_model, DATA_FILE
    train_set = HorseHistoryDataset(DATA_FILE, test=False)
    test_set = HorseHistoryDataset(DATA_FILE, test=True)
    train_model(RNN(train_set.n_features),
                DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn),
                DataLoader(test_set, batch_size=32, collate_fn=collate_fn),
                n_epochs=2)

    from logit_baseline import load_split, race_log_loss, market_log_loss
    X, race, won, odds = load_split(test=True)
    model = torch.nn.Linear(X.shape[1], 1)
    assert torch.isfinite(race_log_loss(model(X).squeeze(1), race, won))
    assert torch.isfinite(market_log_loss(odds, race, won))

    import transformer
    tr_train = transformer.RaceDataset(DATA_FILE, test=False)
    tr_test = transformer.RaceDataset(DATA_FILE, test=True)
    transformer.train_model(
        transformer.RaceTransformer(tr_train.n_features, d_model=16, nhead=2, nlayers=1),
        DataLoader(tr_train, batch_size=8, shuffle=True, collate_fn=transformer.collate_fn),
        DataLoader(tr_test, batch_size=8, collate_fn=transformer.collate_fn),
        n_epochs=2)
    assert torch.isfinite(torch.tensor(transformer.market_log_loss(tr_test)))

    print("OK")


if __name__ == "__main__":
    main()
