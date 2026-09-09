"""Benter second stage: does log(model prob) carry information the market
price does NOT already contain? The model losing to the market standalone
does not answer this — the second stage only needs the model's errors to be
independent of the market's, not smaller.

Honest protocol: the primary model is fit on slice A, its probabilities are
generated OUT OF SAMPLE on slice B, and the blend weights are fit on B. If
the blend were fit on A's in-sample probabilities it would see an
overconfident model and under-weight it. Test is touched once, at the end.
"""
import pandas as pd
import torch
from torch import nn

from rnn import DATA_FILE, META_COLS
from logit_baseline import race_log_loss

df = pd.read_csv(DATA_FILE, index_col=[0])

def pack(d):
    X = torch.tensor(d.drop(columns=META_COLS).values, dtype=torch.float32)
    race = torch.tensor(pd.factorize(d['date_race_id'])[0])
    won = torch.tensor(d['won'].values, dtype=torch.float32)
    odds = torch.tensor(d['odds'].values, dtype=torch.float32)
    return X, race, won, odds

train, test = df[~df.is_test], df[df.is_test]
# temporal split of TRAIN: A fits the primary model, B fits the blend
races = train['date_race_id'].drop_duplicates().sort_values().values
cut = races[int(len(races) * 0.75)]
A, B = train[train.date_race_id < cut], train[train.date_race_id >= cut]
print(f'A(fit) {A.date_race_id.nunique()} races | B(blend) {B.date_race_id.nunique()} '
      f'| test {test.date_race_id.nunique()}')

XA, rA, wA, _ = pack(A)
XB, rB, wB, oB = pack(B)
XT, rT, wT, oT = pack(test)

# --- stage 1: conditional logit on A ---
m = nn.Linear(XA.shape[1], 1)
opt = torch.optim.Adam(m.parameters(), lr=0.05)
for _ in range(200):
    loss = race_log_loss(m(XA).squeeze(1), rA, wA)
    opt.zero_grad(); loss.backward(); opt.step()

def log_probs(scores, race):
    """within-race log softmax"""
    n = int(race.max()) + 1
    mx = torch.full((n,), -torch.inf).scatter_reduce(0, race, scores, reduce='amax')
    se = torch.zeros(n).scatter_add(0, race, (scores - mx[race]).exp())
    return scores - (mx + se.log())[race]

def market_log_probs(odds, race):
    p = 1.0 / odds.clamp(min=1.01)
    n = int(race.max()) + 1
    tot = torch.zeros(n).scatter_add(0, race, p)
    return (p / tot[race]).log()

with torch.no_grad():
    lmB, lmT = log_probs(m(XB).squeeze(1), rB), log_probs(m(XT).squeeze(1), rT)
lqB, lqT = market_log_probs(oB, rB), market_log_probs(oT, rT)

# --- stage 2: blend, fit on B (model probs there are out of sample) ---
blend = nn.Linear(2, 1, bias=False)
with torch.no_grad():                      # start at "market only"
    blend.weight.copy_(torch.tensor([[0.0, 1.0]]))
opt2 = torch.optim.Adam(blend.parameters(), lr=0.02)
featB = torch.stack([lmB, lqB], 1)
for _ in range(400):
    loss = race_log_loss(blend(featB).squeeze(1), rB, wB)
    opt2.zero_grad(); loss.backward(); opt2.step()

w = blend.weight.detach().squeeze()
print(f'\nblend weights: model {w[0]:.3f}, market {w[1]:.3f}')

# CONTROL. The market weight lands above 1.0, which sharpens the market price
# and corrects favourite-longshot bias all on its own. Without this control a
# gain from sharpening is misread as the model adding information.
sharp = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    sharp.weight.copy_(torch.tensor([[1.0]]))
opt3 = torch.optim.Adam(sharp.parameters(), lr=0.02)
for _ in range(400):
    loss = race_log_loss(sharp(lqB[:, None]).squeeze(1), rB, wB)
    opt3.zero_grad(); loss.backward(); opt3.step()
print(f'market-only exponent (control): {sharp.weight.item():.3f}')

with torch.no_grad():
    featT = torch.stack([lmT, lqT], 1)
    res = {
        'model alone   ': race_log_loss(lmT, rT, wT).item(),
        'market alone  ': race_log_loss(lqT, rT, wT).item(),
        'market sharpened (control)': race_log_loss(
            sharp(lqT[:, None]).squeeze(1), rT, wT).item(),
        'model + market': race_log_loss(blend(featT).squeeze(1), rT, wT).item(),
    }
print('\nTEST log-loss:')
for k, v in res.items():
    print(f'  {k}: {v:.4f}')
raw = res['market alone  '] - res['model + market']
ctrl = res['market alone  '] - res['market sharpened (control)']
print(f'\nblend vs market alone      : {raw:+.4f} nats')
print(f'  of which pure sharpening : {ctrl:+.4f} nats  (no model involved)')
print(f'  ATTRIBUTABLE TO THE MODEL: {raw - ctrl:+.4f} nats')


def bootstrap_gap(a_logp, b_logp, race, won, n_boot=2000, seed=0):
    """Paired bootstrap over RACES of the per-race log-loss difference.
    Without this, 'the model adds -0.0001 nats' is indistinguishable from
    'this test cannot resolve anything smaller than its own noise'."""
    g = torch.Generator().manual_seed(seed)
    n = int(race.max()) + 1
    # per-race loss for each scorer
    la = -(a_logp * won)
    lb = -(b_logp * won)
    pa = torch.zeros(n).scatter_add(0, race, la)
    pb = torch.zeros(n).scatter_add(0, race, lb)
    diff = pb - pa                      # >0 means `a` is better
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boots = diff[idx].mean(1)
    return diff.mean().item(), boots.std().item(), \
        torch.quantile(boots, torch.tensor([0.025, 0.975])).tolist()


if __name__ == '__main__':
    with torch.no_grad():
        blended = log_probs(blend(featT).squeeze(1), rT)
        sharpened = log_probs(sharp(lqT[:, None]).squeeze(1), rT)
    print('\n--- is the difference resolvable? paired bootstrap over test races ---')
    for label, cand, base in (
            ('blend vs market      ', blended, lqT),
            ('blend vs sharpened   ', blended, sharpened),
            ('model-in-blend effect', blended, sharpened)):
        m, se, ci = bootstrap_gap(cand, base, rT, wT)
        print(f'  {label}: {m:+.4f} nats  SE {se:.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]')
    print(f'\n  for scale: model-vs-market gap is '
          f'{(race_log_loss(lqT, rT, wT) - race_log_loss(lmT, rT, wT)).abs().item():.4f} nats')
