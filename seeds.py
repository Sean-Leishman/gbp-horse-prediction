"""Train seeds 1 and 2 on the walk-forward folds, for the pre-registered rule.

`PLAN.md` §Pre-registration (vault `67c0538`) needs per-seed blend probabilities
to test `min over seeds {0,1,2} of (p_s - q) > 3%`. Seed 0 is already in
`power.py`'s caches; this fills in 1 and 2 by the identical pipeline with only
`torch.manual_seed` changed.

**This script trains. It does not read the bar.** It writes per-seed
probabilities and nothing else, so it cannot be the 7th read by accident.

Resumable: one cache per (year, seed), skipped if present. ~45 min each, so the
whole thing is ~4.5h and survives being interrupted.

    PYTHONPATH=. python seeds.py
"""
import sys

import torch

from benter_stage2 import (log_probs, market_log_probs, morning_priced, pack,
                           transformer_stage1)
from holdout import fit_blend
from power import CACHE_DIR
from walkforward import FOLDS
from walkforward_clean import fold_data

SEEDS = (1, 2)


def main():
    for year in FOLDS:
        todo = [s for s in SEEDS
                if not (CACHE_DIR / f'seed_cache_{year}_{s}.pt').exists()]
        if not todo:
            print(f'fold {year}: seeds {SEEDS} already cached', flush=True)
            continue

        # Rebuild exactly as power.py does, so row order matches its cache. The
        # assert below is what makes that claim checkable rather than assumed --
        # a silent misalignment would corrupt the read and look like a result.
        ref = torch.load(CACHE_DIR / f'power_cache_{year}.pt')
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

        if not torch.equal(rT, ref['race']) or not torch.allclose(lqT.exp(), ref['q'], atol=1e-6):
            sys.exit(f'{year}: rows do not match power_cache — refusing to write a '
                     f'cache the read would silently misalign')

        for seed in todo:
            # One call: fit_blend needs stage-1 scores on B from the same trained
            # model that scored the test fold, as power.py does.
            torch.manual_seed(seed)
            sB, sT = transformer_stage1(XA, rA, wA, XB, rB, XT, rT)
            blend = fit_blend(log_probs(sB, rB), lqB, rB, wB)
            with torch.no_grad():
                lp = log_probs(blend(torch.stack([log_probs(sT, rT), lqT], 1)).squeeze(1), rT)
            out = CACHE_DIR / f'seed_cache_{year}_{seed}.pt'
            torch.save({'p': lp.exp(), 'q': lqT.exp(), 'race': rT, 'seed': seed,
                        'weights': blend.weight.detach().squeeze().tolist()}, out)
            print(f'fold {year} seed {seed}: wrote {out}  '
                  f'blend weights {[round(x, 3) for x in blend.weight.detach().squeeze().tolist()]}',
                  flush=True)


if __name__ == '__main__':
    main()
