"""Does the model know something the market doesn't? — the question that IS powered.

`power.py` showed the betting test cannot be rescued: at the frozen 3% threshold a
full year of 2016 needs ROI >8.1% to pass, against a +4.57% best estimate.
`concentration.py` showed why no window fixes it — 10 bets of 13,974 carry 82% of
the profit, so the effective sample size is tail events, not rows.

That kills "is it profitable?", not "is there information?". Race-level log-loss is
a DENSE statistic: every race contributes, every runner's probability is scored, and
nothing is thresholded away. n = races, not n = tail events. So the information
question is testable at high power on data already in hand.

Paired bootstrap over races, because blend and market score the SAME races: the
statistic is the per-race difference, which cancels race difficulty.

A win here means the blend carries information the market price does not. It does
NOT mean the strategy is profitable -- that conversion runs through the tail, and
the tail is what we cannot measure. Reporting one as the other would be the same
error as D-007.

    python calibration.py        # seconds: reads power.py's caches
"""
import torch

from power import CACHE_DIR, BOOT
from walkforward import FOLDS, THRESHOLD as BAND_THR


def race_nll(prob, race, won, n_races):
    """-log p(winner) per race. One scatter: exactly one winner per race."""
    return torch.zeros(n_races).scatter_add(0, race[won], -prob[won].clamp(min=1e-12).log())


def implied(odds, race, n_races):
    """BSP-implied probability, normalised within race. The SHARP benchmark: BSP is
    the final price, having absorbed every bet up to the off. Beating morning_wap
    could just be beating a stale quote; beating BSP cannot."""
    raw = 1.0 / odds
    return raw / torch.zeros(n_races).scatter_add(0, race, raw)[race]


def sharpened(q, race, won, n_races, single):
    """D-002's mandatory control: the market with a free exponent, p ~ q**alpha.

    Raising the exponent sharpens the market and improves log-loss on its own, so a
    blend that merely re-sharpens the price reads as "the model adds information".
    Alpha is fitted ON THE TEST FOLD -- deliberately, to hand the control the best
    alpha it could ever have. A blend that still wins has beaten the strongest form
    of the null, so the one-sidedness is conservative, not sloppy.
    """
    lq = q.clamp(min=1e-12).log()
    best = None
    for a in torch.arange(0.70, 2.01, 0.01):
        z = a * lq
        mx = torch.full((n_races,), -1e30).scatter_reduce(0, race, z, 'amax')
        e = (z - mx[race]).exp()
        nll = race_nll(e / torch.zeros(n_races).scatter_add(0, race, e)[race],
                       race, won, n_races)[single]
        if best is None or nll.mean() < best[1]:
            best = (a.item(), nll.mean().item(), nll)
    return best


def paired_ci(diff):
    """Bootstrap the MEAN per-race difference. Paired, so race difficulty cancels."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(diff), (BOOT, len(diff)), generator=g)
    boot = diff[idx].mean(1)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()
    return lo, hi


def main():
    print(f'{"fold":>6} {"races":>6} {"mornWAP":>8} {"BSP":>8} {"blend":>8} '
          f'{"vs morn":>8}  95% CI of gain')
    all_diff, vs_bsp, vs_sharp, alphas = [], [], [], []
    for year in FOLDS:
        d = torch.load(CACHE_DIR / f'power_cache_{year}.pt')
        race, n = d['race'], d['test_races']
        won = d['pnl'] > 0
        # Multinomial log-loss needs exactly one winner. Dead heats pay two (half
        # stake at full odds each), so they have no single winner and are dropped --
        # 0.2% of races, and walkforward.py already drops them from training.
        wins = torch.zeros(n).scatter_add(0, race[won], torch.ones(int(won.sum())))
        assert (wins <= 2).all() and (wins >= 1).all(), f'{year}: winner count off'
        single = wins == 1

        mkt = race_nll(d['q'], race, won, n)[single]
        bsp = race_nll(implied(d['odds'], race, n), race, won, n)[single]
        bld = race_nll(d['p'], race, won, n)[single]
        diff = mkt - bld                      # positive = blend beats morning market
        vs_bsp.append(bsp - bld)
        alpha, _, sharp = sharpened(d['q'], race, won, n, single)
        vs_sharp.append(sharp - bld)
        alphas.append(alpha)
        n = int(single.sum())
        all_diff.append(diff)
        lo, hi = paired_ci(diff)
        print(f'{year:>6} {n:>6} {mkt.mean():>8.4f} {bsp.mean():>8.4f} '
              f'{bld.mean():>8.4f} {diff.mean():>+8.4f}  [{lo:+.4f}, {hi:+.4f}]')

    diff = torch.cat(all_diff)
    lo, hi = paired_ci(diff)
    verdict = 'INFORMATIVE' if lo > 0 else 'not distinguishable from the market'
    print(f'\nPOOLED {len(diff)} races: gain {diff.mean():+.4f} nats/race '
          f'[{lo:+.4f}, {hi:+.4f}]')
    print(f'  {verdict}')
    print(f'  Minimum detectable gain at this n: {(hi - lo) / 2:.4f} nats/race.')
    # D-002: the gain is not quotable without this line.
    sh = torch.cat(vs_sharp)
    slo, shi = paired_ci(sh)
    print(f'vs best-possible sharpened market (alpha {min(alphas):.2f}-{max(alphas):.2f}, '
          f'fitted on test): {sh.mean():+.4f} [{slo:+.4f}, {shi:+.4f}]')
    print(f'  {"CONTROL PASSED -- the model adds information" if slo > 0 else
             "CONTROL FAILED -- this is market sharpening, not the model"}')

    b = torch.cat(vs_bsp)
    blo, bhi = paired_ci(b)
    print(f'vs BSP (the final price): {b.mean():+.4f} nats/race [{blo:+.4f}, {bhi:+.4f}]'
          f' -- {"still ahead" if blo > 0 else "BEHIND or tied"}')
    # Why losing to BSP matters so much here: the bet DECIDES on morning_wap but is
    # MATCHED at BSP (D-005). So we never hold the stale price we thought was wrong --
    # we accept the final one. If the market systematically corrects toward our view
    # by the off, the edge we selected on is gone before the bet exists.
    print(f'\n{"thr":>5} {"bets":>7} {"model p":>8} {"morn q":>8} {"BSP":>8} '
          f'{"actual":>8}  price drift')
    for thr in (0.0, 0.01, 0.03):
        dr, beats = [], []
        for year, d in zip(FOLDS, (torch.load(CACHE_DIR / f'power_cache_{y}.pt')
                                   for y in FOLDS)):
            race, n = d['race'], d['test_races']
            b = implied(d['odds'], race, n)
            sel = d['edge'] > thr
            dr.append(torch.stack([d['p'][sel], d['q'][sel], b[sel],
                                   (d['pnl'][sel] > 0).float()]))
        x = torch.cat(dr, 1)
        pm, qm, bm, act = x.mean(1).tolist()
        print(f'{thr:>5.0%} {x.shape[1]:>7} {pm:>8.4f} {qm:>8.4f} {bm:>8.4f} '
              f'{act:>8.4f}  {(bm - qm) / qm:+.1%} toward {"us" if bm > qm else "against us"}')
    print('  model p vs actual is the model\'s own calibration on what it chose to back;')
    print('  BSP vs morn q is how far the market moved after we decided and before we paid.')

    # The band table is the one that decides whether the edge is exploitable. The
    # pooled +1.1pp could be ten lucky longshots (D-008) or a real effect spread
    # across prices. Sign and size per band tell them apart, and the model's own
    # p sits next to the truth so its calibration error is visible where the money is.
    print('\nEDGE AT THE PRICE WE PAY (actual strike - BSP implied), by BSP band')
    print(f'{"band":>10} {"bets":>6} {"BSP imp":>8} {"actual":>8} {"edge":>8} {"+-1SE":>7} '
          f'{"model p":>8} {"ROI":>8} {"P&L share":>9}')
    cols = []
    for year in FOLDS:
        d = torch.load(CACHE_DIR / f'power_cache_{year}.pt')
        race, nr = d['race'], d['test_races']
        sel = d['edge'] > BAND_THR
        cols.append(torch.stack([implied(d['odds'], race, nr)[sel],
                                 (d['pnl'][sel] > 0).float(), d['odds'][sel],
                                 d['pnl'][sel], d['p'][sel]]))
    bsp_s, won_s, odds_s, pnl_s, p_s = torch.cat(cols, 1)
    tot = pnl_s.sum()
    for lo_o, hi_o in ((1, 3), (3, 6), (6, 11), (11, 21), (21, 1e9)):
        m = (odds_s > lo_o) & (odds_s <= hi_o)
        k = int(m.sum())
        if not k:
            continue
        a, bi = won_s[m].mean().item(), bsp_s[m].mean().item()
        lab = f'{lo_o}-{hi_o:g}' if hi_o < 1e9 else f'{lo_o}+'
        print(f'{lab:>10} {k:>6} {bi:>8.4f} {a:>8.4f} {a - bi:>+8.4f} '
              f'{(a * (1 - a) / k) ** .5:>7.4f} {p_s[m].mean():>8.4f} '
              f'{pnl_s[m].mean() * 100:>+7.2f}% {pnl_s[m].sum() / tot * 100:>8.1f}%')
    a, bi = won_s.mean().item(), bsp_s.mean().item()
    print(f'{"ALL":>10} {len(won_s):>6} {bi:>8.4f} {a:>8.4f} {a - bi:>+8.4f} '
          f'{(a * (1 - a) / len(won_s)) ** .5:>7.4f} {p_s.mean():>8.4f} '
          f'{pnl_s.mean() * 100:>+7.2f}% {100.0:>8.1f}%')

    print('\nThis is an information claim, not a profit claim. A positive gain here is '
          'consistent with\nthe unprofitable betting result: log-loss is dense, P&L is '
          'tail-driven, and D-008 applies.')


if __name__ == '__main__':
    main()
