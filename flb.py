"""Favourite-longshot bias on margin-free BSP. The second stage wanted a
market exponent of 1.15 (sharpen the price), which implies favourites are
underbet. This measures it directly, and prices it against costs -- BSP is
takeable (Betfair SP betting), so an edge here is real money or nothing.
"""
import pandas as pd, numpy as np
YEARS = range(2009, 2014)          # backfilled, real BSP
df = pd.concat(pd.read_csv(
    f'/home/seanleishman/Projects/rpscrape/data/region/gb/flat/{y}.csv', dtype=str)
    for y in YEARS)
num = lambda s: pd.to_numeric(s, errors='coerce')
d = df.assign(bsp=num(df.bsp), won=(df.pos.str.strip() == '1').astype(int))
d = d[d.bsp > 1.0].dropna(subset=['bsp'])
# normalise within race so implied probs sum to 1 (BSP overround is ~0.3%)
d['imp'] = 1 / d.bsp
d['imp'] /= d.groupby('race_id').imp.transform('sum')

bands = [1, 2, 3, 4, 6, 8, 12, 20, 40, 1000]
d['band'] = pd.cut(d.bsp, bands)
g = d.groupby('band', observed=True).agg(
    n=('won', 'size'), implied=('imp', 'mean'), actual=('won', 'mean'),
    mean_bsp=('bsp', 'mean'))
g['edge_pp'] = (g.actual - g.implied) * 100
# flat-stake back return at BSP, 5% commission on wins (Betfair standard)
g['back_roi_%'] = ((g.actual * (g.mean_bsp - 1) * 0.95) - (1 - g.actual)) * 100
print(f'{len(d)} runners, {d.race_id.nunique()} races, {min(YEARS)}-{max(YEARS)}\n')
print(g.round(4).to_string())
print('\nback-the-favourite (shortest BSP in each race), flat stakes, 5% comm:')
fav = d.loc[d.groupby('race_id').bsp.idxmin()]
roi = ((fav.won * (fav.bsp - 1) * 0.95) - (1 - fav.won)).mean()
se = ((fav.won * (fav.bsp - 1) * 0.95) - (1 - fav.won)).std() / np.sqrt(len(fav))
print(f'  n={len(fav)}  strike {fav.won.mean()*100:.1f}%  '
      f'ROI {roi*100:+.2f}%  (SE {se*100:.2f}pp, t={roi/se:+.2f})')
