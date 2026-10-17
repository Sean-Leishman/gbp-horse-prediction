"""Where does the flat-stake edge actually come from?

Run after `power.py` (it reads that script's fold tensor caches). Answers the
question that the staking comparison in `power.py` raised: equal-profit staking
cuts the CI width 2-3x and takes the ROI to zero with it, which can only happen
if the profit lives at long prices.

It does. At the pre-registered 3% threshold the 93% of bets priced under 11.0
contribute 3.6% of the profit; the top 10 bets out of 13,974 supply 82%. That is
the real reason no window length fixes the power: the EFFECTIVE sample size is
the number of profit-carrying events (dozens), not the number of bets.

Diagnostic only -- 2011-2013 is spent evidence. Nothing here selects anything.
"""
import torch
from walkforward import FOLDS

for thr in (0.02, 0.03):
    print(f'\n=== threshold {thr:.0%} — flat stakes, pooled 2011-2013 ===')
    print(f'{"BSP band":>12} {"bets":>7} {"strike":>7} {"ROI":>8} {"share of total P&L":>20}')
    e, pnl, odds = [], [], []
    for y in FOLDS:
        f = torch.load(f'data/power_cache_{y}.pt')
        m = f['edge'] > thr
        e.append(f['edge'][m]); pnl.append(f['pnl'][m]); odds.append(f['odds'][m])
    pnl, odds = torch.cat(pnl), torch.cat(odds)
    total = pnl.sum()
    print(f'{"ALL":>12} {len(pnl):>7} {"":>7} {pnl.mean()*100:>+7.2f}% {total:>19.0f}u')
    for lo, hi in ((1, 3), (3, 6), (6, 11), (11, 21), (21, 1e9)):
        b = (odds > lo) & (odds <= hi)
        if b.sum() < 20:
            continue
        lbl = f'{lo}-{hi:g}' if hi < 1e8 else f'{lo}+'
        print(f'{lbl:>12} {int(b.sum()):>7} {"":>7} {pnl[b].mean()*100:>+7.2f}% '
              f'{pnl[b].sum()/total*100:>18.1f}%')
    # how much rests on the single best result?
    top = pnl.sort(descending=True).values
    print(f'  top 10 bets supply {top[:10].sum()/total*100:.1f}% of all profit; '
          f'top 50 supply {top[:50].sum()/total*100:.1f}%')
    print(f'  without the single best bet, ROI = {(pnl.sum()-top[0])/len(pnl)*100:+.2f}%')
