"""How large an ROI could the 2016 holdout actually detect?

`holdout.py` pre-registers the bar as "ROI positive with a 95% CI excluding
zero". Nothing was ever checked about whether a single year of GB flat racing
can PRODUCE such a CI. That is a power question, and it is answerable without
the holdout data: a walk-forward fold's test year is 6,030-6,395 races, and the
2016 holdout window (2016-01-15 onward) is ~6,130 — the same instrument.

So: retrain each fold under the holdout's own protocol, and for each threshold
measure the bootstrap CI half-width at fold size and at the three candidate
holdout windows. Because the bar is "CI excludes zero", the half-width IS the
minimum detectable ROI: any true edge smaller than it cannot pass, however real.

The ROI column here is SPENT EVIDENCE — 2011-2013 were evaluated by
`walkforward_clean.py` on 2026-09-30. It is printed for context only. If the
pre-registered threshold is ever changed, the defensible basis for the new value
is the WIDTH column (a variance property of the instrument), never the ROI one.

    python power.py          # ~20 min/fold; fold data is cached
"""
import torch

from benter_stage2 import (COMMISSION, log_probs, market_log_probs, pack,
                           morning_priced, settle, transformer_stage1)
from holdout import fit_blend
from walkforward import FOLDS, SEED
from walkforward_clean import fold_data

THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.03)
BOOT = 2000

# GB flat races per month, 2015 (the shape 2016 is assumed to repeat). The
# holdout window opens 2016-01-15, so January counts 17/31 of its races.
RACES_2015 = (277, 248, 260, 521, 689, 791, 844, 869, 676, 528, 278, 276)
JAN = round(RACES_2015[0] * 17 / 31)
WINDOWS = {
    'banked Jan15-Mar31': JAN + sum(RACES_2015[1:3]),
    'H1  Jan15-Jun30':    JAN + sum(RACES_2015[1:6]),
    'FULL Jan15-Dec31':   JAN + sum(RACES_2015[1:]),
    '2016+2017 (2 yrs)': (JAN + sum(RACES_2015[1:])) + sum(RACES_2015),
}


def ci(pnl_r, stk_r, n_draw=None):
    """Bootstrap over bet-carrying races. n_draw < len() simulates a SHORTER
    window drawn from the same P&L distribution; the point estimate is unchanged
    by that, only the width is, which is exactly the quantity in question."""
    n = len(pnl_r) if n_draw is None else n_draw
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(pnl_r), (BOOT, n), generator=g)
    boot = pnl_r[idx].sum(1) / stk_r[idx].sum(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
    return lo, hi, (hi - lo) / 2


def main():
    folds = {}
    for year in FOLDS:
        df = fold_data(year)
        train = morning_priced(df[~df.is_test])
        test = morning_priced(df[df.is_test & (df.date <= f'{year}-12-31')])
        races = train['date_race_id'].drop_duplicates().sort_values().values
        cut = races[int(len(races) * 0.75)]
        A, B = train[train.date_race_id < cut], train[train.date_race_id >= cut]

        XA, rA, wA, _ = pack(A)
        XB, rB, wB, _ = pack(B)
        XT, rT, wT, oT = pack(test)
        lqB = market_log_probs(torch.tensor(B['morning_wap'].values, dtype=torch.float32), rB)
        lqT = market_log_probs(torch.tensor(test['morning_wap'].values, dtype=torch.float32), rT)

        torch.manual_seed(SEED)
        sB, sT = transformer_stage1(XA, rA, wA, XB, rB, XT, rT)
        blend = fit_blend(log_probs(sB, rB), lqB, rB, wB)
        with torch.no_grad():
            lp = log_probs(blend(torch.stack([log_probs(sT, rT), lqT], 1)).squeeze(1), rT)
        folds[year] = (lp.exp() - lqT.exp(), settle(wT, oT, rT), rT,
                       test['date_race_id'].nunique())
        print(f'fold {year}: test {folds[year][3]} races', flush=True)

    n_races = {y: int(r.max()) + 1 for y, (_, _, r, _) in folds.items()}
    print(f'\n{"=" * 78}\nPER-FOLD (one year each — the holdout is one year)\n{"=" * 78}')
    print(f'{"thr":>5} {"year":>5} {"bets":>6} {"ROI":>8} {"95% CI":>20} {"+/-":>7}')
    pooled = {}
    for thr in THRESHOLDS:
        agg = []
        for year, (edge, pnl, race, _) in folds.items():
            sel = edge > thr
            pr = torch.zeros(n_races[year]).scatter_add(0, race, pnl * sel)
            sr = torch.zeros(n_races[year]).scatter_add(0, race, sel.float())
            act = sr > 0
            pr, sr = pr[act], sr[act]
            agg.append((pr, sr))
            lo, hi, hw = ci(pr, sr)
            print(f'{thr:>5.1%} {year:>5} {int(sr.sum()):>6} '
                  f'{pr.sum() / sr.sum() * 100:>+7.2f}% '
                  f'[{lo * 100:>+6.2f}%,{hi * 100:>+6.2f}%] {hw * 100:>+6.2f}pp')
        pooled[thr] = (torch.cat([a for a, _ in agg]), torch.cat([b for _, b in agg]))

    print(f'\n{"=" * 78}\nMINIMUM DETECTABLE ROI by window (= CI half-width; '
          f'below this, no pass is possible)\n{"=" * 78}')
    hdr = ''.join(f'{k:>22}' for k in WINDOWS)
    print(f'{"thr":>5} {"bets/race":>10}{hdr}')
    for thr, (pr, sr) in pooled.items():
        per_race = len(pr) / sum(f[3] for f in folds.values())   # bet-carrying fraction
        row = ''
        for races in WINDOWS.values():
            n = max(30, round(per_race * races))
            _, _, hw = ci(pr, sr, n)
            row += f'{f"{int(n * (sr.sum() / len(sr)))} bets  >{hw * 100:.1f}%":>22}'
        print(f'{thr:>5.1%} {per_race:>10.2f}{row}')
    print(f'\nBar is "CI excludes zero" at {COMMISSION:.0%} commission. A window whose '
          'minimum\ndetectable ROI exceeds the plausible edge (+1.4% to +7.9% observed) '
          'cannot\nreturn a pass, and its failure would carry no information.')


if __name__ == '__main__':
    main()
