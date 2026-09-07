"""Is `draws` carrying draw bias, or just field size?"""
import pandas as pd, numpy as np
df = pd.concat(pd.read_csv(
    f'/home/seanleishman/Projects/rpscrape/data/region/gb/flat/{y}.csv', dtype=str)
    for y in range(2008, 2016))
num = lambda s: pd.to_numeric(s, errors='coerce')
d = df.assign(draw=num(df.draw), won=(df.pos.str.strip() == '1').astype(int))
d = d[d.draw > 0].dropna(subset=['draw'])
d['field'] = d.groupby('race_id').draw.transform('size')

print('win rate by FIELD SIZE bucket (the confound):')
d['fb'] = pd.cut(d.field, [0, 6, 8, 10, 12, 16, 40])
print('  ', {str(k): round(v, 4) for k, v in d.groupby('fb', observed=True).won.mean().items()})

print('\nwin rate by draw percentile WITHIN ITS OWN RACE (field-size neutral):')
d['pct'] = d.groupby('race_id').draw.rank(pct=True)
d['pd'] = (d.pct * 5).clip(0, 4.999).astype(int)
print('  ', {int(k): round(v, 4) for k, v in d.groupby('pd').won.mean().items()})

print('\nsame, split by field size — draw bias should bite hardest in big fields:')
for lab, sub in d.groupby(pd.cut(d.field, [0, 8, 12, 40]), observed=True):
    r = sub.groupby('pd').won.mean()
    print(f'  field {str(lab):10s} n={len(sub):6d}: ',
          {int(k): round(v, 4) for k, v in r.items()})

print('\nlow vs high draw at the courses with the strongest split (>=3000 runners):')
big = d[d.field >= 10]
g = big.groupby('course').apply(
    lambda s: pd.Series({'n': len(s),
                         'low': s[s.pct <= .33].won.mean(),
                         'high': s[s.pct >= .67].won.mean()}), include_groups=False)
g = g[g.n >= 3000]
g['edge'] = g.low - g.high
print(g.reindex(g.edge.abs().sort_values(ascending=False).index).head(8).round(4).to_string())
