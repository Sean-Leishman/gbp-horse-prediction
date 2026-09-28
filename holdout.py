"""PRE-REGISTERED out-of-time holdout. Written 2026-09-16, BEFORE the data it
needs exists (the scrape is mid-2016). Nothing here may be tuned once the data
lands — that is the point of committing it early.

The 2026-09-15 result (+7.9%/+7.7%/+3.2% over three seeds) came from a test
window that had by then been looked at across three decision-price variants.
This spends a window that has never been looked at, once.

PROTOCOL (fixed):
  * holdout   = races on/after HOLDOUT_START. 2016-01-01..14 are excluded
                because a stale partial 2016.csv put them in the window already
                examined on 2026-09-15.
  * train     = everything before HOLDOUT_START; slice A (first 75% of races)
                fits stage 1, slice B (last 25%) fits the blend. Same as the gate.
  * stage 1   = set-transformer, seeds 0, 1, 2. The logit already failed
                (+0.04%, CI [-4.0, +3.8]) and is reported only for reference.
  * decision  = morning_wap, settlement = BSP, 5% commission (D-005).
  * threshold = FIXED AT 3%. No selection on the holdout, none on B.
  * statistic = the three seeds as one portfolio: stake 1/3 on each seed's
                selections, bootstrap over holdout races.
  * BAR       = portfolio ROI positive with a 95% CI excluding zero.
                Anything else is a fail, including "positive at another
                threshold" or "two seeds out of three".

    python holdout.py            # transformer seeds 0,1,2 (~30 min each)
    python holdout.py --logit    # reference only
"""
import sys

import pandas as pd
import torch

from benter_stage2 import (COMMISSION, log_probs, logit_stage1, market_log_probs,
                           morning_priced, pack, settle, transformer_stage1)
from logit_baseline import race_log_loss
from rnn import DATA_FILE
from torch import nn

HOLDOUT_START = '2016-01-15'
HOLDOUT_END = '2016-12-31'   # the window is fixed at both ends: a later year
                             # must not widen it silently once it is scraped
THRESHOLD = 0.03
SEEDS = (0, 1, 2)


def fit_blend(lm, lq, race, won):
    blend = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        blend.weight.copy_(torch.tensor([[0.0, 1.0]]))    # start at market-only
    opt = torch.optim.Adam(blend.parameters(), lr=0.02)
    feat = torch.stack([lm, lq], 1)
    for _ in range(400):
        loss = race_log_loss(blend(feat).squeeze(1), race, won)
        opt.zero_grad(); loss.backward(); opt.step()
    return blend


def main():
    use_logit = '--logit' in sys.argv
    df = pd.read_csv(DATA_FILE, index_col=[0])
    train = df[df.date < HOLDOUT_START]
    hold = df[(df.date >= HOLDOUT_START) & (df.date <= HOLDOUT_END)]
    train, hold = morning_priced(train), morning_priced(hold)
    if not len(hold):
        sys.exit(f'no morning-priced races on/after {HOLDOUT_START} yet — '
                 'the scrape has not reached them')
    # Spend the holdout ONCE, on the finished window. Running it as months
    # accumulate is several looks at the test set, which is the whole thing
    # pre-registering it was meant to stop -- so this is a check, not a note
    # in the log. December racing exists every year; if the window does not
    # reach it, the scrape is unfinished.
    if hold.date.max() < HOLDOUT_END[:4] + '-12-01' and '--force-partial' not in sys.argv:
        sys.exit(f'holdout window stops at {hold.date.max()} — the scrape has not '
                 f'finished {HOLDOUT_END[:4]}. Wait for it; do not spend the '
                 'holdout on a partial year (--force-partial to override).')
    races = train['date_race_id'].drop_duplicates().sort_values().values
    cut = races[int(len(races) * 0.75)]
    A, B = train[train.date_race_id < cut], train[train.date_race_id >= cut]
    print(f'A(fit) {A.date_race_id.nunique()} races | B(blend) {B.date_race_id.nunique()} | '
          f'holdout {hold.date_race_id.nunique()} races, {hold.date.min()}..{hold.date.max()}')
    assert hold.date.min() >= HOLDOUT_START

    XA, rA, wA, _ = pack(A)
    XB, rB, wB, oB = pack(B)
    XH, rH, wH, oH = pack(hold)
    dB = torch.tensor(B['morning_wap'].values, dtype=torch.float32)
    dH = torch.tensor(hold['morning_wap'].values, dtype=torch.float32)
    lqB, lqH = market_log_probs(dB, rB), market_log_probs(dH, rH)
    pnl_unit = settle(wH, oH, rH)
    n_races = int(rH.max()) + 1

    per_seed = []
    for seed in (0,) if use_logit else SEEDS:
        torch.manual_seed(seed)
        sB, sH = (logit_stage1(XA, rA, wA, XB, XH) if use_logit else
                  transformer_stage1(XA, rA, wA, XB, rB, XH, rH))
        blend = fit_blend(log_probs(sB, rB), lqB, rB, wB)
        with torch.no_grad():
            lp = log_probs(blend(torch.stack([log_probs(sH, rH), lqH], 1)).squeeze(1), rH)
        sel = (lp.exp() - lqH.exp()) > THRESHOLD
        print(f'  seed {seed}: {int(sel.sum())} bets  strike {wH[sel].mean()*100:.2f}%  '
              f'ROI {pnl_unit[sel].mean()*100:+.2f}%'
              f'  (blend weights {blend.weight.detach().squeeze().tolist()})')
        per_seed.append(sel.float())

    # One portfolio: 1/3 of a unit on each seed's selections. Per-race P&L and
    # per-race stake, so the bootstrap keeps correlated bets in a race together.
    stake = torch.stack(per_seed).mean(0)
    pnl_r = torch.zeros(n_races).scatter_add(0, rH, stake * pnl_unit)
    stk_r = torch.zeros(n_races).scatter_add(0, rH, stake)
    active = stk_r > 0
    pnl_r, stk_r = pnl_r[active], stk_r[active]
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(pnl_r), (2000, len(pnl_r)), generator=g)
    boot = pnl_r[idx].sum(1) / stk_r[idx].sum(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
    roi = pnl_r.sum() / stk_r.sum()
    print(f'\nPORTFOLIO (edge>{THRESHOLD:.0%}, {COMMISSION:.0%} commission, '
          f'{int(active.sum())} races, {stake.sum():.0f} unit-stakes)')
    print(f'  ROI {roi*100:+.2f}%  95% CI [{lo*100:+.2f}%, {hi*100:+.2f}%]')
    print(f'  {"PASS" if lo > 0 else "FAIL"} — bar was: positive with CI excluding zero')


if __name__ == '__main__':
    main()
