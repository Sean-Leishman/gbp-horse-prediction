"""Does a price-dependent shrink fix the model's overconfidence? Pre-registered.

The protocol is frozen in `PLAN.md` §Pre-registration (vault commit b49e3ca,
written before this file existed) and is NOT restated here as an invitation to
adjust it. In short: the model overstates win probability by +1.55pp against a
+1.11pp edge, 2.1x at BSP 21+, so no staking rule is safe on raw `p`.

    p_cal  ~  exp( lam(q)*log p + (1 - lam(q))*log q ),  lam(q) = a + b*log q

renormalised within race. lam shrinks the model's DISAGREEMENT with the market
toward the market; letting lam depend on log q lets it shrink more at long prices,
where the error is. a=1, b=0 is today's model, so the fit cannot help by
construction and the read is a real test. A global lam is pointless -- fit_blend
already optimises it -- so price-dependence is the entire hypothesis.

Fit on fold 2011's test year, read on 2012+2013 pooled (n = 12,245 races). The
objective is race log-loss, deliberately NOT the bar: matching the objective to
the bar would make the bar circular.

ROI is printed as context and is not part of the bar. ROI/MDE = 0.56 here (D-007)
and its effective sample size is tail events (D-008). It decides nothing.

    python recalibrate.py          # seconds: reads power.py's caches
"""
import torch

from calibration import implied, paired_ci, race_nll
from power import CACHE_DIR
from walkforward import THRESHOLD

FIT_FOLD, READ_FOLDS = '2011', ('2012', '2013')


def load(year):
    d = torch.load(CACHE_DIR / f'power_cache_{year}.pt')
    race, n = d['race'], d['test_races']
    won = d['pnl'] > 0
    wins = torch.zeros(n).scatter_add(0, race[won], torch.ones(int(won.sum())))
    return d, race, n, won, wins == 1          # dead heats dropped, as in calibration.py


def apply_cal(p, q, race, n, a, b):
    """lam(q) = a + b*log q, applied in log space and renormalised within race."""
    lp, lq = p.clamp(min=1e-12).log(), q.clamp(min=1e-12).log()
    lam = a + b * lq
    z = lam * lp + (1 - lam) * lq
    mx = torch.full((n,), -1e30).scatter_reduce(0, race, z.detach(), 'amax')
    e = (z - mx[race]).exp()
    return e / torch.zeros(n).scatter_add(0, race, e)[race]


def fit():
    d, race, n, won, single = load(FIT_FOLD)
    ab = torch.tensor([1.0, 0.0], requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        pc = apply_cal(d['p'], d['q'], race, n, ab[0], ab[1])
        loss = race_nll(pc, race, won, n)[single].mean()
        loss.backward()
        return loss

    base = race_nll(d['p'], race, won, n)[single].mean().item()
    opt.step(closure)
    a, b = ab.detach().tolist()
    with torch.no_grad():
        got = race_nll(apply_cal(d['p'], d['q'], race, n, ab[0], ab[1]),
                       race, won, n)[single].mean().item()
    print(f'FIT on {FIT_FOLD}: a={a:+.4f} b={b:+.4f}  '
          f'(a=1,b=0 is today\'s model)\n'
          f'  race log-loss {base:.4f} -> {got:.4f} in sample\n'
          f'  lam at q=0.50 {a + b * torch.tensor(0.50).log():.3f}, '
          f'at q=0.10 {a + b * torch.tensor(0.10).log():.3f}, '
          f'at q=0.03 {a + b * torch.tensor(0.03).log():.3f}')
    return a, b


def main():
    a, b = fit()

    mkt_n, old_n, new_n, rows = [], [], [], []
    for year in READ_FOLDS:
        d, race, n, won, single = load(year)
        with torch.no_grad():
            pc = apply_cal(d['p'], d['q'], race, n,
                           torch.tensor(a), torch.tensor(b))
        mkt_n.append(race_nll(d['q'], race, won, n)[single])
        old_n.append(race_nll(d['p'], race, won, n)[single])
        new_n.append(race_nll(pc, race, won, n)[single])
        rows.append((d, race, n, won, pc, implied(d['odds'], race, n)))

    mkt, old, new = torch.cat(mkt_n), torch.cat(old_n), torch.cat(new_n)
    print(f'\nREAD on {"+".join(READ_FOLDS)}, {len(mkt)} races — one read, '
          f'no variants after it\n' + '=' * 66)

    # --- Bar 2: no loss of information ---
    lo, hi = paired_ci(mkt - new)
    dlo, dhi = paired_ci(old - new)
    print(f'BAR 2  log-loss  market {mkt.mean():.4f}  old {old.mean():.4f}  '
          f'new {new.mean():.4f}')
    print(f'       new vs market {(mkt - new).mean():+.4f} [{lo:+.4f}, {hi:+.4f}]'
          f'  {"positive, CI excludes zero" if lo > 0 else "FAILS"}')
    print(f'       new vs old    {(old - new).mean():+.4f} [{dlo:+.4f}, {dhi:+.4f}]'
          f'  {"not worse" if dhi > 0 or dlo > -1e-4 else "WORSE"}')
    bar2 = lo > 0 and (old - new).mean() > -1e-4

    # --- Bar 1: calibration at the committed selection threshold ---
    print(f'\nBAR 1  calibration at the committed {THRESHOLD:.0%} threshold')
    print(f'{"selection":>22} {"bets":>7} {"mean prob":>10} {"actual":>8} '
          f'{"error":>8} {"ROI":>9}')
    bar1 = None
    for lab, prob, edge in (('old  p - q  (before)', 'p', None),
                            ('new  pcal - q (BAR)', 'cal', 'cal')):
        bets, pm, act, pnl = [], [], [], []
        for d, race, n, won, pc, _ in rows:
            pr = pc if prob == 'cal' else d['p']
            sel = (pr - d['q']) > THRESHOLD
            bets.append(int(sel.sum()))
            pm.append(pr[sel]); act.append(won[sel].float()); pnl.append(d['pnl'][sel])
        pm, act, pnl = torch.cat(pm), torch.cat(act), torch.cat(pnl)
        err = (pm.mean() - act.mean()).item()
        print(f'{lab:>22} {sum(bets):>7} {pm.mean():>10.4f} {act.mean():>8.4f} '
              f'{err:>+8.4f} {pnl.mean() * 100:>+8.2f}%')
        if prob == 'cal':
            bar1 = abs(err) <= 0.005

    print(f'\n  BAR 1 calibration within +-0.5pp: {"PASS" if bar1 else "FAIL"}')
    print(f'  BAR 2 information retained:       {"PASS" if bar2 else "FAIL"}')
    if not bar1:
        print('\n  Per the pre-registration, a BAR 1 failure means the overconfidence is a')
        print('  WINNER\'S CURSE ON SELECTION, not miscalibration: we pick the horses whose')
        print('  p is highest partly because its error is largest there. The fix would be a')
        print('  selection rule that accounts for estimation error, not a calibration curve.')
        print('  That is a result. No second attempt on these folds.')
    print('\n  ROI above is context, not a bar (D-007: ROI/MDE = 0.56; D-008: tail-driven).')


if __name__ == '__main__':
    main()
