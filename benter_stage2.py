"""Benter second stage: does log(model prob) carry information the market
price does NOT already contain? A model losing to the market standalone does
not answer this — the blend only needs the model's errors to be independent of
the market's, not smaller.

    python benter_stage2.py              # stage 1 = conditional logit
    python benter_stage2.py --transformer # stage 1 = set-transformer

Protocol (all of it load-bearing — see DECISIONS.md D-002):
  * the primary model is fit on slice A and its probabilities generated OUT OF
    SAMPLE on slice B; the blend weights are fit on B. Fitting the blend on A's
    in-sample probabilities would show it an overconfident model and
    under-weight it.
  * a market-sharpening CONTROL: the blend can beat the market by raising its
    exponent above 1 alone, which reads as "the model adds information". It
    does not. Against bookmaker odds this control is worth +0.0037 nats.
  * a paired bootstrap over races, so "about zero" is a measurement.
Test is touched once, at the end.
"""
import sys

import pandas as pd
import torch
from torch import nn

from rnn import DATA_FILE, META_COLS
from logit_baseline import race_log_loss


def pack(d):
    X = torch.tensor(d.drop(columns=META_COLS).values, dtype=torch.float32)
    race = torch.tensor(pd.factorize(d['date_race_id'])[0])
    won = torch.tensor(d['won'].values, dtype=torch.float32)
    odds = torch.tensor(d['odds'].values, dtype=torch.float32)
    return X, race, won, odds


def log_probs(scores, race):
    """Within-race log softmax."""
    n = int(race.max()) + 1
    mx = torch.full((n,), -torch.inf).scatter_reduce(0, race, scores, reduce='amax')
    se = torch.zeros(n).scatter_add(0, race, (scores - mx[race]).exp())
    return scores - (mx + se.log())[race]


def market_log_probs(odds, race):
    p = 1.0 / odds.clamp(min=1.01)
    n = int(race.max()) + 1
    tot = torch.zeros(n).scatter_add(0, race, p)
    return (p / tot[race]).log()


# ---------------------------------------------------------------- stage 1 --

def logit_stage1(XA, rA, wA, XB, XT):
    m = nn.Linear(XA.shape[1], 1)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(200):
        loss = race_log_loss(m(XA).squeeze(1), rA, wA)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return m(XB).squeeze(1), m(XT).squeeze(1)


def as_races(X, race):
    """Flat rows -> [n_races, max_runners, F] + pad mask, plus the index map
    needed to scatter per-runner scores back into flat row order."""
    n = int(race.max()) + 1
    counts = torch.bincount(race, minlength=n)
    order = torch.argsort(race, stable=True)
    r = race[order]
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(race)) - starts[r]
    padded = torch.zeros(n, int(counts.max()), X.shape[1])
    padded[r, pos] = X[order]
    mask = torch.ones(n, int(counts.max()), dtype=torch.bool)
    mask[r, pos] = False
    return padded, mask, order, r, pos


def transformer_stage1(XA, rA, wA, XB, rB, XT, rT, epochs=20, bs=64):
    """Same set-transformer as transformer.py, trained on slice A with early
    stopping on a temporal validation slice carved from A's own tail."""
    from transformer import RaceTransformer
    import copy

    pA, mA, oA, rrA, poA = as_races(XA, rA)
    # index of the winning runner within each padded race row
    hit = wA[oA].bool()
    w_pos = torch.zeros(pA.shape[0], dtype=torch.long)
    w_pos[rrA[hit]] = poA[hit]

    # Dead heats (2 winners) break a single-label cross-entropy. 111 of 50,011
    # races; transformer.py's own dataset skips them too. Excluded from
    # TRAINING only — scoring B/test needs no winner, so the gate still runs on
    # exactly the same races the logit path used.
    ok = (torch.bincount(rrA[hit], minlength=pA.shape[0]) == 1).nonzero().squeeze(1)
    assert not mA[ok, w_pos[ok]].any(), 'winner landed on a pad'
    n_val = len(ok) // 10                     # temporal: race ids are date-ordered
    fit_idx, val_idx = ok[:len(ok) - n_val], ok[len(ok) - n_val:]
    print(f'  stage1 train {len(fit_idx)} races, val {len(val_idx)} '
          f'({pA.shape[0] - len(ok)} dead heats dropped)', flush=True)

    model = RaceTransformer(XA.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state = float('inf'), None
    for ep in range(epochs):
        model.train()
        perm = fit_idx[torch.randperm(len(fit_idx))]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = nn.functional.cross_entropy(model(pA[b], mA[b]), w_pos[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = sum(nn.functional.cross_entropy(
                model(pA[val_idx[i:i + bs]], mA[val_idx[i:i + bs]]),
                w_pos[val_idx[i:i + bs]], reduction='sum').item()
                for i in range(0, len(val_idx), bs)) / len(val_idx)
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
        print(f'  stage1 epoch {ep}: val {vl:.4f}{"  <- best" if best == vl else ""}',
              flush=True)
    model.load_state_dict(best_state)

    def score(X, race):
        p, m, o, r, po = as_races(X, race)
        model.eval()
        out = torch.empty(len(race))
        with torch.no_grad():
            for i in range(0, p.shape[0], bs):
                s = model(p[i:i + bs], m[i:i + bs])
                sel = (r >= i) & (r < i + bs)
                out[o[sel]] = s[r[sel] - i, po[sel]]
        return out

    return score(XB, rB), score(XT, rT)


# ------------------------------------------------------------------ gate ---

def bootstrap_gap(a_logp, b_logp, race, won, n_boot=2000, seed=0):
    """Paired bootstrap over RACES. Without it, '-0.0001 nats' cannot be told
    apart from 'this test resolves nothing smaller than its own noise'."""
    g = torch.Generator().manual_seed(seed)
    n = int(race.max()) + 1
    pa = torch.zeros(n).scatter_add(0, race, -(a_logp * won))
    pb = torch.zeros(n).scatter_add(0, race, -(b_logp * won))
    diff = pb - pa
    boots = diff[torch.randint(0, n, (n_boot, n), generator=g)].mean(1)
    return diff.mean().item(), boots.std().item(), \
        torch.quantile(boots, torch.tensor([0.025, 0.975])).tolist()


def run_gate(sB, sT, rB, wB, oB, rT, wT, oT):
    lmB, lmT = log_probs(sB, rB), log_probs(sT, rT)
    lqB, lqT = market_log_probs(oB, rB), market_log_probs(oT, rT)

    blend = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        blend.weight.copy_(torch.tensor([[0.0, 1.0]]))    # start at market-only
    opt = torch.optim.Adam(blend.parameters(), lr=0.02)
    featB = torch.stack([lmB, lqB], 1)
    for _ in range(400):
        loss = race_log_loss(blend(featB).squeeze(1), rB, wB)
        opt.zero_grad(); loss.backward(); opt.step()

    sharp = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        sharp.weight.copy_(torch.tensor([[1.0]]))
    opt3 = torch.optim.Adam(sharp.parameters(), lr=0.02)
    for _ in range(400):
        loss = race_log_loss(sharp(lqB[:, None]).squeeze(1), rB, wB)
        opt3.zero_grad(); loss.backward(); opt3.step()

    w = blend.weight.detach().squeeze()
    print(f'\nblend weights: model {w[0]:.3f}, market {w[1]:.3f}')
    print(f'market-only exponent (control): {sharp.weight.item():.3f}')

    with torch.no_grad():
        blended = log_probs(blend(torch.stack([lmT, lqT], 1)).squeeze(1), rT)
        sharpened = log_probs(sharp(lqT[:, None]).squeeze(1), rT)
    res = {
        'model alone              ': race_log_loss(lmT, rT, wT).item(),
        'market alone             ': race_log_loss(lqT, rT, wT).item(),
        'market sharpened (control)': race_log_loss(sharpened, rT, wT).item(),
        'model + market           ': race_log_loss(blended, rT, wT).item(),
    }
    print('\nTEST log-loss:')
    for k, v in res.items():
        print(f'  {k}: {v:.4f}')
    raw = res['market alone             '] - res['model + market           ']
    ctrl = res['market alone             '] - res['market sharpened (control)']
    print(f'\nblend vs market alone      : {raw:+.4f} nats')
    print(f'  of which pure sharpening : {ctrl:+.4f} nats  (no model involved)')
    print(f'  ATTRIBUTABLE TO THE MODEL: {raw - ctrl:+.4f} nats')

    m, se, ci = bootstrap_gap(blended, sharpened, rT, wT)
    print(f'\n  model-in-blend effect: {m:+.4f} nats  SE {se:.4f}  '
          f'95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]')
    print(f'  {"RESOLVABLY NON-ZERO" if ci[0] > 0 else "consistent with ZERO"}'
          f'  (model-vs-market gap for scale: '
          f'{res["model alone              "] - res["market alone             "]:.4f})')


def main():
    use_tf = '--transformer' in sys.argv
    df = pd.read_csv(DATA_FILE, index_col=[0])
    train, test = df[~df.is_test], df[df.is_test]
    races = train['date_race_id'].drop_duplicates().sort_values().values
    cut = races[int(len(races) * 0.75)]
    A, B = train[train.date_race_id < cut], train[train.date_race_id >= cut]
    print(f'stage 1 = {"set-transformer" if use_tf else "conditional logit"}')
    print(f'A(fit) {A.date_race_id.nunique()} races | B(blend) '
          f'{B.date_race_id.nunique()} | test {test.date_race_id.nunique()}')

    XA, rA, wA, _ = pack(A)
    XB, rB, wB, oB = pack(B)
    XT, rT, wT, oT = pack(test)
    if use_tf:
        sB, sT = transformer_stage1(XA, rA, wA, XB, rB, XT, rT)
    else:
        sB, sT = logit_stage1(XA, rA, wA, XB, XT)
    run_gate(sB, sT, rB, wB, oB, rT, wT, oT)


if __name__ == '__main__':
    main()
