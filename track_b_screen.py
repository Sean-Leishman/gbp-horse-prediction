"""Track B screening: pre-off momentum in Betfair GB horse WIN markets.

Basic-tier data (1-min last-traded-price, no volume). Question: do runners
whose price shortens in the signal window (T-30m..T-10m) keep shortening in
the outcome window (T-10m..off)?  Reports rank correlation and the gross
return of "back the biggest steamer at T-10, settle at off LTP".

Usage: python track_b_screen.py data/betfair/BASIC/2024/Oct [more dirs...]
"""

import bz2
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIGNAL_START = 30 * 60_000  # ms before off
SIGNAL_END = 10 * 60_000
MIN_PRE_OFF_POINTS = 3  # per runner, inside the 30-min window


def parse_market(path):
    """Return per-runner (p_start, p_signal_end, p_off) for GB WIN markets, else None."""
    try:
        with bz2.open(path, "rt") as f:
            first = json.loads(f.readline())
            md = first["mc"][0].get("marketDefinition")
            if (
                md is None
                or md.get("marketType") != "WIN"
                or md.get("countryCode") != "GB"
                or md.get("eventTypeId") != "7"
                or not md.get("bspMarket")  # real day-of-race markets are BSP markets
            ):
                return None

            ltp = {}  # runner_id -> [(pt, price)]
            off_pt = None
            for line in f:
                msg = json.loads(line)
                for mc in msg["mc"]:
                    d = mc.get("marketDefinition")
                    if d and d.get("inPlay") and off_pt is None:
                        off_pt = msg["pt"]
                    for rc in mc.get("rc", []):
                        ltp.setdefault(rc["id"], []).append((msg["pt"], rc["ltp"]))
                if off_pt is not None:
                    break  # pre-off only; ignore in-play tail

        if off_pt is None or not ltp:
            return None

        out = []
        for rid, series in ltp.items():
            pre = [(pt, p) for pt, p in series if pt <= off_pt]
            sig = [p for pt, p in pre if off_pt - pt >= SIGNAL_END and off_pt - pt <= SIGNAL_START]
            outw = [p for pt, p in pre if off_pt - pt < SIGNAL_END]
            if len(sig) < MIN_PRE_OFF_POINTS or not outw:
                continue
            out.append((sig[0], sig[-1], outw[-1]))
        return out if len(out) >= 4 else None
    except Exception:
        return None  # ponytail: skip unparseable/truncated files, count via None


def main(dirs):
    files = [p for d in dirs for p in Path(d).rglob("1.*.bz2")]
    print(f"{len(files)} market files")

    with Pool() as pool:
        results = pool.map(parse_market, files, chunksize=64)

    markets = [r for r in results if r]
    rows = np.array([t for m in markets for t in m])
    print(f"{len(markets)} GB WIN markets parsed, {len(rows)} runner series")
    if not len(rows):
        return

    p0, p1, poff = rows[:, 0], rows[:, 1], rows[:, 2]
    sig_move = np.log(p1 / p0)   # <0 = steamed (shortened) in signal window
    out_move = np.log(poff / p1)  # <0 = kept shortening

    from scipy.stats import spearmanr

    rho, pval = spearmanr(sig_move, out_move)
    print(f"\nSpearman(signal move, outcome move): rho={rho:.4f}  p={pval:.2e}")

    # Trading rule: back the biggest steamer per market at T-10 LTP, settle at off LTP.
    # Gross return per unit stake = p1/poff - 1 (back low... backing at price p1,
    # laying off at poff: profit ratio approx p1/poff - 1 for small moves).
    idx = 0
    rets = []
    for m in markets:
        a = np.array(m)
        moves = np.log(a[:, 1] / a[:, 0])
        i = moves.argmin()
        if moves[i] < 0:  # only bet if something actually steamed
            rets.append(a[i, 1] / a[i, 2] - 1)
        idx += len(m)
    rets = np.array(rets)
    print(f"\nBack-the-steamer at T-10, settle at off: {len(rets)} bets")
    print(f"mean gross return {rets.mean()*100:.3f}%  (t={rets.mean()/rets.std()*np.sqrt(len(rets)):.2f})")
    print(f"median {np.median(rets)*100:.3f}%  win-rate {(rets>0).mean()*100:.1f}%")
    print("\nCosts to beat: ~1 tick spread crossing each way (~0.5-1%) ; 2% commission on net winnings.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/betfair/BASIC"])
