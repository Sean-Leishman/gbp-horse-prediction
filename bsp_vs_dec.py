"""D-001 evidence: does margin-free BSP actually beat bookmaker `dec` as the
market benchmark? No — within-race normalisation cancels a proportional
overround, so the two log-losses agree to ~0.002 nats (vs a 0.18-nat model
gap). Run this before ever re-gating model work on BSP coverage.
"""
import glob, numpy as np, pandas as pd
df = pd.concat(pd.read_csv(f'/home/seanleishman/Projects/rpscrape/data/region/gb/flat/{y}.csv', dtype=str) for y in (2009, 2010))
num = lambda s: pd.to_numeric(s, errors='coerce')
df = df.assign(bsp=num(df.bsp), dec=num(df.dec), won=(df.pos.str.strip() == '1').astype(int))
df = df.dropna(subset=['bsp', 'dec'])
df = df[(df.bsp > 1.0) & (df.dec > 1.0)]
# only races where every runner has both prices and exactly one winner
ok = df.groupby('race_id').won.transform('sum').eq(1)
df = df[ok]

def market_ll(odds, race, won):
    p = 1.0 / odds
    tot = pd.Series(p).groupby(race).transform('sum').values
    return -np.log(np.clip((p / tot)[won == 1], 1e-12, None)).mean()

race, won = df.race_id.values, df.won.values
print(f'{len(df)} runners, {df.race_id.nunique()} races')
print(f'overround (dec): {(1/df.dec).groupby(df.race_id).sum().mean():.3f}  '
      f'(bsp): {(1/df.bsp).groupby(df.race_id).sum().mean():.3f}')
for name, col in (('BSP ', df.bsp.values), ('dec ', df.dec.values)):
    print(f'{name}market log-loss: {market_ll(col, race, won):.4f}')
