"""The pre-registered edge-reliability read. ONE RUN. Do not run until seeds are done.

Protocol frozen in `PLAN.md` §Pre-registration (vault `67c0538`, refinement
pinned before the seed caches existed). Written before the data, deliberately, so
the implementation cannot drift toward a result.

    select iff  min over seeds {0,1,2} of (p_s - q) > 3%

against a control: seed-0's single-seed threshold raised until it selects the
SAME number of bets. The matched count is the whole point -- per D-011, a bar
against zero pays a rule for pruning, since the MDE rises from 0.72pp at 13,974
bets to 1.44pp at 3,500 against a defect of 1.55pp.

Both arms score with the SEED-AVERAGED probability, so only selection differs.
The pure status quo (seed-0 p, seed-0 selection) prints as context.

This is the 7th read of these folds and is SELECTIVE, NOT CONFIRMATORY: seven
reads at 95% carry a ~30% family-wise error rate, so it can choose a rule and
cannot establish one. 2016 is the only unread data.

    PYTHONPATH=. python reliability.py
"""
import sys

import torch

from calibration import paired_ci, race_nll
from power import CACHE_DIR, BOOT
from walkforward import FOLDS, THRESHOLD

SEEDS = (0, 1, 2)
MIN_BETS = 5000          # below this the read is underpowered; see PLAN.md


def fold(year):
    ref = torch.load(CACHE_DIR / f'power_cache_{year}.pt')
    ps = [ref['p']]
    for s in SEEDS[1:]:
        f = CACHE_DIR / f'seed_cache_{year}_{s}.pt'
        if not f.exists():
            sys.exit(f'{f} missing — seeds.py has not finished. Not reading a '
                     f'partial portfolio.')
        d = torch.load(f)
        if not torch.equal(d['race'], ref['race']) or not torch.allclose(d['q'], ref['q'], atol=1e-6):
            sys.exit(f'{f}: rows misaligned with power_cache — refusing to read')
        ps.append(d['p'])
    race, n = ref['race'], ref['test_races']
    won = ref['pnl'] > 0
    wins = torch.zeros(n).scatter_add(0, race[won], torch.ones(int(won.sum())))
    return ref, race, n, won, wins == 1, torch.stack(ps)


def err_by_race(sel, prob, won, race, n):
    """Per-race (sum of prob, count of wins, count of bets) so the bootstrap can
    resample races and recompute the SAME statistic the point estimate reports."""
    f = sel.float()
    return (torch.zeros(n).scatter_add(0, race, prob * f),
            torch.zeros(n).scatter_add(0, race, won.float() * f),
            torch.zeros(n).scatter_add(0, race, f))


def main():
    P, Q, WON, RACE, OFF, SINGLE, MKT, AVG0 = [], [], [], [], 0, [], [], []
    for year in FOLDS:
        ref, race, n, won, single, ps = fold(year)
        P.append(ps); Q.append(ref['q']); WON.append(won); RACE.append(race + OFF)
        SINGLE.append(single)
        MKT.append(race_nll(ref['q'], race, won, n)[single])
        AVG0.append(race_nll(ref['p'], race, won, n)[single])
        OFF += n
    ps = torch.cat(P, 1); q = torch.cat(Q); won = torch.cat(WON); race = torch.cat(RACE)
    n = OFF
    pbar = ps.mean(0)
    edge_min = (ps - q).min(0).values
    sel_b = edge_min > THRESHOLD
    nb = int(sel_b.sum())

    print(f'{len(FOLDS)} folds, {n} races, seeds {SEEDS}')
    print(f'\nPOWER CHECK (read before the bar, per PLAN.md)')
    print(f'  status quo, seed-0 edge > {THRESHOLD:.0%}: {int(((ps[0] - q) > THRESHOLD).sum())} bets')
    print(f'  unanimity rule:                  {nb} bets '
          f'({nb / int(((ps[0] - q) > THRESHOLD).sum()) * 100:.0f}% retained)')
    ok = nb >= MIN_BETS
    print(f'  {"ADEQUATELY POWERED" if ok else f"UNDERPOWERED (<{MIN_BETS} bets)"}'
          f' — {"" if ok else "the bar below must be reported as such, not interpreted"}')

    # Matched-count control: raise seed-0's threshold until it selects nb bets.
    e0 = ps[0] - q
    thr = torch.sort(e0, descending=True).values[nb - 1].item()
    sel_c = e0 >= thr
    print(f'\nMATCHED CONTROL: seed-0 threshold raised {THRESHOLD:.2%} -> {thr:.2%} '
          f'to select {int(sel_c.sum())} bets')

    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, n, (BOOT, n), generator=g)

    def arm(sel, prob, label):
        sp, sw, sn = err_by_race(sel, prob, won, race, n)
        pt = (sp.sum() / sn.sum() - sw.sum() / sn.sum()).item()
        boot = (sp[idx].sum(1) / sn[idx].sum(1)) - (sw[idx].sum(1) / sn[idx].sum(1))
        lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
        print(f'  {label:<34} {int(sn.sum()):>6} bets  '
              f'prob {(sp.sum()/sn.sum()):.4f}  actual {(sw.sum()/sn.sum()):.4f}  '
              f'error {pt*100:+.2f}pp [{lo*100:+.2f}, {hi*100:+.2f}]')
        return boot, pt

    print(f'\nBAR 1 (primary) — same probability in both arms, only selection differs')
    bb, pb = arm(sel_b, pbar, 'unanimity (seed-avg prob)')
    bc, pc = arm(sel_c, pbar, 'matched control (seed-avg prob)')
    _, _ = arm((ps[0] - q) > THRESHOLD, ps[0], 'status quo (seed-0, context only)')

    d = bb - bc
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975])).tolist()
    print(f'\n  difference (unanimity - control): {(pb - pc)*100:+.2f}pp '
          f'[{lo*100:+.2f}, {hi*100:+.2f}]')
    bar1 = (pb - pc) < 0 and hi < 0
    print(f'  BAR 1: {"PASS" if bar1 else "FAIL"} — negative with CI excluding zero')

    mkt, a0 = torch.cat(MKT), torch.cat(AVG0)
    single = torch.cat(SINGLE)
    navg = race_nll(pbar, race, won, n)[single]
    lo2, hi2 = paired_ci(a0 - navg)
    bar2 = (a0 - navg).mean().item() > -1e-4
    print(f'\nBAR 2 (dense) — seed-avg log-loss {navg.mean():.4f} vs seed-0 {a0.mean():.4f}')
    print(f'  gain {(a0 - navg).mean():+.4f} [{lo2:+.4f}, {hi2:+.4f}]  '
          f'{"not worse" if bar2 else "WORSE"}')
    print(f'  BAR 2: {"PASS" if bar2 else "FAIL"}')

    print(f'\nSELECTIVE, NOT CONFIRMATORY (7th read of these folds). '
          f'{"A pass chooses a rule for 2016; it does not establish one." if bar1 else ""}')
    if not ok:
        print('AND UNDERPOWERED — the bar reading above is not interpretable.')


if __name__ == '__main__':
    main()
