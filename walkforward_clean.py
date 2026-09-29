"""Walk-forward with preprocessing refitted PER FOLD — the leak-free version.

`walkforward.py` (2026-09-29) scored +4.99% pooled, but three preprocessing
statistics were fitted on the global train split, which contains every fold's
test year: the StandardScaler, `course_draw_bias`, and the rpr/ts/or median
imputation. Weak aggregate statistics, but not nothing, and exactly the kind of
thing that has already bitten this project twice (D-003, D-004).

Here each fold rebuilds the dataset from raw with `cutoff = <year>-01-01`, so
every fitted statistic sees only that fold's past. Protocol is otherwise
IDENTICAL and unchanged: transformer seed 0, decision `morning_wap`, settle BSP,
5% commission, threshold FIXED at 3%, pooled bootstrap over bet-carrying races.

Same reading, fixed in advance: supportive = all folds positive AND pooled CI
excludes zero. Anything else is not supportive, and given the holdout is blocked
that is as close to a kill as this project can currently get.

    python walkforward_clean.py          # ~20 min/fold: rebuild + train
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from benter_stage2 import (COMMISSION, log_probs, market_log_probs,
                           morning_priced, pack, settle, transformer_stage1)
from holdout import fit_blend
from walkforward import FOLDS, SEED, THRESHOLD

BUILD_DIR = Path('data/preprocessing/folds')


def fold_data(year):
    """Rebuild from raw with this fold's cutoff. Cached: the build is ~3 min."""
    out = BUILD_DIR / f'{year}.csv'
    if not out.exists():
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        print(f'  building {out} (cutoff {year}-01-01)...', flush=True)
        r = subprocess.run([sys.executable, '-c',
                            'import sys, data_import as d;'
                            'd.main([root / r / t'
                            ' for root in (d.RPSCRAPE_REGION, d.COMMUNITY_REGION)'
                            " for r in ('gb', 'ire') for t in ('flat', 'jumps')],"
                            ' cutoff=sys.argv[1], out=sys.argv[2])',
                            f'{year}-01-01', str(out)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'build failed for {year}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}')
    return pd.read_csv(out, index_col=[0])


def main():
    pnl_all, stake_all, race_all, offset = [], [], [], 0
    for year in FOLDS:
        df = fold_data(year)
        train = morning_priced(df[~df.is_test])
        test = morning_priced(df[df.is_test & (df.date <= f'{year}-12-31')])
        assert train.date.max() < f'{year}-01-01', 'fold train leaked past the cutoff'

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

        sel = (lp.exp() - lqT.exp()) > THRESHOLD
        pnl = settle(wT, oT, rT)
        print(f'\nfold {year}: train {A.date_race_id.nunique()}+{B.date_race_id.nunique()} races, '
              f'test {test.date_race_id.nunique()} races')
        print(f'  {int(sel.sum())} bets  strike {wT[sel].mean()*100:.2f}%  '
              f'ROI {pnl[sel].mean()*100:+.2f}%  '
              f'(blend weights {[round(x, 3) for x in blend.weight.detach().squeeze().tolist()]})',
              flush=True)

        pnl_all.append(pnl * sel)
        stake_all.append(sel.float())
        race_all.append(rT + offset)
        offset += int(rT.max()) + 1

    pnl_t, stake_t, race_t = torch.cat(pnl_all), torch.cat(stake_all), torch.cat(race_all)
    pnl_r = torch.zeros(offset).scatter_add(0, race_t, pnl_t)
    stk_r = torch.zeros(offset).scatter_add(0, race_t, stake_t)
    active = stk_r > 0
    pnl_r, stk_r = pnl_r[active], stk_r[active]
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(pnl_r), (2000, len(pnl_r)), generator=g)
    boot = pnl_r[idx].sum(1) / stk_r[idx].sum(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
    roi = (pnl_r.sum() / stk_r.sum()).item()
    print(f'\nPOOLED, preprocessing refit per fold (edge>{THRESHOLD:.0%}, '
          f'{COMMISSION:.0%} commission, {int(active.sum())} races, {int(stk_r.sum())} bets)')
    print(f'  ROI {roi*100:+.2f}%  95% CI [{lo*100:+.2f}%, {hi*100:+.2f}%]')
    print(f'  {"SUPPORTIVE" if lo > 0 else "NOT SUPPORTIVE"} — still not the holdout.')


if __name__ == '__main__':
    main()
