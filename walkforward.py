"""PRE-REGISTERED walk-forward. Written and committed 2026-09-29 BEFORE running.

WHY THIS EXISTS, AND WHAT IT IS NOT. The out-of-time holdout (`holdout.py`) is
blocked indefinitely: Racing Post IP-banned this machine, so 2016+ cannot be
scraped. This is the best *available* substitute, and it is weaker.

  * The examined test window starts **2014-06-06**. Every year before that has
    only ever been TRAINING data — no ROI number has ever been computed on it,
    and no model was ever selected on it. So as evaluation periods, 2011/2012/
    2013 are fresh.
  * They are NOT pristine. The pipeline and its fixes (D-003's banned flags,
    D-004's same-day leak) were found using data that spans these years, so the
    feature set is not innocent of them. A true out-of-time holdout still
    outranks this, and `holdout.py` stays unrun and pre-registered for the day
    the data question is resolved.
  * Earlier folds train on less history (2011 gets ~19k races vs ~45k for the
    2016 holdout), so the models are weaker. That biases AGAINST finding an
    edge, which is the safe direction to be wrong in.

PROTOCOL (fixed, not to be edited after seeing output):
  * folds      = test years 2011, 2012, 2013, each trained on every race
                 strictly before 1 January of that year; A = first 75% of those
                 races (stage 1), B = last 25% (blend weights).
  * stage 1    = set-transformer, seed 0 (one seed per fold — the 2026-09-15
                 seed spread was in threshold selection, and the threshold is
                 fixed here).
  * decision   = morning_wap, settlement = BSP, 5% commission (D-005).
  * threshold  = FIXED AT 3%. No selection anywhere.
  * statistic  = per-fold ROI, plus a pooled bootstrap over all bet-carrying
                 races across folds.
  * READING IT (stated in advance):
      - supportive  = all three folds positive AND pooled CI excludes zero
      - not supportive = anything else. A pooled CI containing zero means the
        2026-09-15 +7.9% is not reproducible across periods, which — with the
        holdout unavailable — is as close to a kill as this project can get
        without new data.

    python walkforward.py
"""
import sys

import pandas as pd
import torch

from benter_stage2 import (COMMISSION, log_probs, market_log_probs,
                           morning_priced, pack, settle, transformer_stage1)
from holdout import fit_blend
from rnn import DATA_FILE

FOLDS = ('2011', '2012', '2013')
THRESHOLD = 0.03
SEED = 0


def main():
    df = pd.read_csv(DATA_FILE, index_col=[0])
    examined = df.loc[df.is_test, 'date'].min()
    pnl_all, stake_all, race_all, offset = [], [], [], 0

    for year in FOLDS:
        assert year < examined[:4], f'{year} is inside the examined window ({examined})'
        train = morning_priced(df[df.date < f'{year}-01-01'])
        test = morning_priced(df[(df.date >= f'{year}-01-01') & (df.date <= f'{year}-12-31')])
        races = train['date_race_id'].drop_duplicates().sort_values().values
        cut = races[int(len(races) * 0.75)]
        A, B = train[train.date_race_id < cut], train[train.date_race_id >= cut]

        XA, rA, wA, _ = pack(A)
        XB, rB, wB, _ = pack(B)
        XT, rT, wT, oT = pack(test)
        dB = torch.tensor(B['morning_wap'].values, dtype=torch.float32)
        dT = torch.tensor(test['morning_wap'].values, dtype=torch.float32)
        lqB, lqT = market_log_probs(dB, rB), market_log_probs(dT, rT)

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
              f'(blend weights {[round(x, 3) for x in blend.weight.detach().squeeze().tolist()]})')

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
    print(f'\nPOOLED over {len(FOLDS)} folds (edge>{THRESHOLD:.0%}, {COMMISSION:.0%} commission, '
          f'{int(active.sum())} races, {int(stk_r.sum())} bets)')
    print(f'  ROI {roi*100:+.2f}%  95% CI [{lo*100:+.2f}%, {hi*100:+.2f}%]')
    print(f'  {"SUPPORTIVE" if lo > 0 else "NOT SUPPORTIVE"} — and this is not the holdout; '
          'it cannot license a bet, only inform the decision.')


if __name__ == '__main__':
    main()
