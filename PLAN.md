# Plan: data rebuild + model evaluation

Goal: predict UK race winners, measured against the market (odds-implied
log-loss). Models are ready and verified on synthetic data (`test_pipeline.py`);
the blocker is raw data — the original CSVs are lost locally.

## Current state (2026-07-11)

- **Pipeline** (`preprocessing.py`): leak-free feature build — past-only rolling
  form, jockey/trainer/pedigree win %, Elo ratings; temporal 80/20 split;
  scaler fit on train only. Outputs `data/preprocessing/6-model-data.csv`.
- **Models**: `logit_baseline.py` (Benter-style conditional logit — the floor),
  `transformer.py` (set-transformer, attention across runners in a race),
  `rnn.py` (LSTM over horse history). All report test log-loss vs the market's
  odds-implied log-loss on identical races.
- **Data**: `data/raw/{races_UK2,runners_UK2,full_data4}.csv` missing. Not in
  `~/wsl-extract` (searched exhaustively). Candidates: `//pop-os` CIFS shares
  (host offline), exFAT USB `5CA7-F1ED` (unplugged).
- **Research verdict**: conditional logit + engineered features is the proven
  approach (Bolton & Chapman 1986, Benter 1994). Transformer evidence exists
  only for attention-across-runners + ratings (CUHK LYU2102). Skip
  history-sequence transformers. UK market (Betfair) is near-efficient; the
  bar is model log-loss < market log-loss, then a Benter second-stage
  (combine model prob with odds-implied prob) to extract edge.

## Phase 0 — recover original data (opportunistic, non-blocking)

- [ ] When `pop-os` is on the network: check `/mnt/backups`, `/mnt/disk`,
      `/mnt/shared` for `races_UK2.csv` / `runners_UK2.csv` / `full_data4.csv`
- [ ] When exFAT drive (UUID `5CA7-F1ED`) is plugged in: check `/mnt/data`
- Old data ends ~2021 — useful as cross-check only; rescrape supersedes it.

## Phase 1 — rescrape via rpscrape

[rpscrape](https://github.com/joenano/rpscrape) is actively maintained
(pushed May 2026). Retires `scraper.py` entirely — Selenium not needed; the
horse-history features it scraped are now computed in preprocessing.

- [ ] USER: create free Racing Post account, grab email + access token from
      browser cookies (per rpscrape README) into its `.env`
- [ ] Clone rpscrape, smoke-test on one day's results
- [ ] Scrape GB + IRE, flat + jumps, 2008–present (matches Betfair SP window)
- [ ] Write `data_import.py`: map rpscrape columns → pipeline raw schema
      (races: id/class/distance/going/type/handicap/date; runners: horse/
      jockey/trainer ids, draw, age, weight, TS/RPR/OR, odds, position;
      pedigree: sire/dam/damsire; beaten lengths)
- [ ] Run full pipeline + all three models on real data; record log-losses

## Phase 2 — Betfair SP as market benchmark

Daily BSP CSVs are free at `promo.betfair.com/betfairsp/prices` (verified
live), May 2008–present: BSP, pre-off WAP, win/lose per runner.

- [ ] Downloader for daily UK win files (resumable, rate-limited)
- [ ] Join to runners on date + course + horse name (watch name normalisation:
      country suffixes, punctuation)
- [ ] Replace bookmaker SP with BSP in the `odds` benchmark column
- [ ] Re-report market log-loss vs models against the margin-free price

## Phase 3 — only if models beat the market log-loss

- [ ] Benter second-stage: small logit combining log(model prob) +
      log(BSP-implied prob) — this is where edge becomes a betting strategy
- [ ] Flat-stakes + fractional-Kelly ROI simulation on the temporal test set
- [ ] GBM (LightGBM) comparison on the same features (Lessmann 2010: trees
      beat logit on HK data)

## Phase 4 — expansion candidates (only after Phase 3 verdict)

Edge = (data/model others lack) × (liquidity) ÷ (quants present). Ranked:

1. **Place / each-way markets** — Harville/Henery reduction over our win probs;
   Betfair place markets less efficient (recreational each-way flow). Zero new
   data, ~50 lines on top of Phase 3. Also Tote place/exacta pools (betting
   against casual pool money, not sharps).
2. **Sectional times (pre-race)** — Total Performance Data covers most UK
   courses since ~2018; pace/finishing-speed figures still barely used in UK
   form (standard in HK/AUS). Most credible "data others ignore" play.
   Check access terms; coverage starts 2018.
3. **Cricket (T20 in-play)** — best new liquid market: Cricsheet ball-by-ball
   data is free and complete, state-space modelling (resources/matchups)
   rewards effort, IPL/international Betfair liquidity is enormous, thin
   modelling community vs football.
4. **Darts** — throw-by-throw stochastic process, academically modellable,
   PDC events UK-liquid on Betfair, few quants. Cricket-like profile, smaller.
5. **Dota 2 (esports)** — genuinely soft (patches shift meta faster than
   bookies reprice), OpenDota/Steam data free. Bet at Pinnacle (doesn't ban
   winners, modest limits). Ceiling: side-income — thin liquidity, and tier-2/3
   match-fixing means adverse selection on "mispriced" lines. CS2 data is
   gatekept by GRID (commercial); LoL via Riot API.
6. **Tennis** — point data free (Jeff Sackmann), but the Markov model is the
   most-published sports model there is; pre-match efficient, in-play is a
   latency/courtsider game. Learning exercise only.

Skip: football majors (xG commoditised, deepest quant pool), golf (DataGolf
sells the edge publicly), player props (bookies restrict winners in weeks),
harness pools ATG/PMU (soft but high access/language friction).

### In-play racing / live feeds — expensive escalation, not a start

- TPD sells a live GPS feed (sub-second positions/sectionals) — the real
  in-play data source, but B2B-priced (likely five figures/yr; email to
  confirm). Retail streams run 5–30s behind; without TPD-tier latency the
  in-play market is unwinnable (on-course + licensed-pictures players own it).
- Only revisit if the Phase 3 pre-race model beats BSP and turnover justifies
  the data cost. Historical TPD sectionals (item 2) capture most of the
  informational edge without the latency war.

### Constraint on everything above

Winners get restricted by bookmakers within weeks → exchange-only → Betfair
Premium Charge takes 20% (up to 40–60%) of consistent winnings. Haircut all
ROI simulations ~20% before believing them. Liquidity outside headline
meetings/events is thinner than it looks.

### Market-entry playbook — same gates for every candidate, kill at first fail

1. **Data gate**: point-in-time reconstructable history (no hindsight fields),
   free or cheap. No point-in-time data → no honest backtest → kill.
2. **Access gate**: where does money actually go down at target stakes?
   Exchange depth, commission, API. Betfair API-NG: delayed key free, live key
   £299 one-off, bots explicitly allowed. Pinnacle is B2B-only now — retail
   route is a broker (Sportmarket / BetInAsia / Mollybet); brokers also
   sidestep bookie restrictions for football/AH.
3. **Model gate**: reuse the horse harness — leak-free features, temporal
   split, log-loss vs the market's closing price. That benchmark generalises
   to every sport: the close is always the thing to beat.
4. **Backtest gate**: Benter-style blend with the market price, then net of
   commission + premium-charge haircut, fractional Kelly. Edge < costs → kill.
5. **CLV gate**: paper-trade live 4–8 weeks measuring closing line value, not
   P&L (CLV converges in weeks; P&L is noise for months). Consistently beat
   the close → go live.
6. **Live**: ≤ 1/4 Kelly, log every bet with the closing price, monitor
   calibration drift — all these markets are non-stationary (patches, rule
   changes, going-report methodology, roster moves).

### Other considerations / ideas

- **Trading, not predicting**: promoted to Track B below.
- **Other racing jurisdictions**: HK — Benter's ground, pools now razor-sharp,
  skip. Japan — huge JRA pools, JRA-VAN data service, language barrier keeps
  anglophone quants out (a moat, if we cross it). Australia — good data
  (TPD-style), Betfair AU thinner than UK.
- **CLV in the harness early**: build closing-price capture into every
  scraper/logger from day one; it's the only fast unbiased edge signal.
- **Own market impact**: in thin markets (darts, Dota, place markets at small
  meetings) our own stake moves the price — measure edge net of impact, cap
  stake per market rather than per day.
- **Overfitting discipline**: freeze one final held-out season per sport,
  untouched until a strategy is locked; walk-forward everything else. One
  peek and it's spent.
- **UK tax/legal**: individual gambling winnings are tax-free; Betfair
  permits API bots. Keep full records regardless (premium-charge tracking
  needs them anyway).

### Edge classes beyond prediction (ranked)

1. **Rollover pools** — Placepot/Scoop6/Tote jackpots/Colossus: ~25–30%
   takeout normally kills them, but rollovers/guarantees inject dead money
   → +EV before skill. Phase 3 win probs → permutation construction makes
   us sharp money in a casual pool. Pools never ban winners. Best synergy
   with the model; the classic syndicate play.
2. **BOG + extra places** — Best Odds Guaranteed = free option on max(taken
   price, SP); +EV on likely drifters (Track B identifies them) with no
   winner prediction needed. Extra-place each-way on big handicaps likewise.
   Near risk-free bankroll-builder; ceiling = account longevity (weeks–months
   per bookie). Fastest real money on this list, capped low.
3. **Event-driven automation** — non-runners, overnight going changes,
   Rule 4 windows: prices mechanically stale for minutes. Bot on the
   declarations/going feeds; reuses Track B API infra. Small, real.
4. **Market-making** — earn the spread in mid-liquidity markets (place,
   small meetings); profit from flow not prediction. Different skillset
   (queue position, adverse selection, inventory); competing with
   established bots. Later, maybe — after Track B teaches microstructure.
5. **Micro-biases** (draw/rail movements, first-time headgear, trainer
   intent) — not standalone edges; fold into Phase 1 features.

## Track B — pre-off price trading (parallel track; needs NO Racing Post data)

Predict Betfair price *moves* in the pre-off window, not race outcomes.
Feedback loop of minutes; data is exchange-only, so this is not blocked on
the Phase 1 rescrape and can start now.

Data facts (verified 2026-07-11, historicdata.betfair.com): all Exchange
markets since 2016; **Basic tier is free** but gives only last-traded price
per minute, no volume, no ladder; **Pro tier** (paid) is 50ms granularity
with volume. Basic = screening only (can't simulate fills without volume).

- [ ] USER: Betfair account → download Basic files, GB horse WIN markets,
      one recent month, from historicdata.betfair.com
- [ ] Parse with `betfairlightweight`/`flumine` (standard libs; they read the
      historic stream JSON natively — do not write a parser)
- [ ] Exploratory: is there momentum? Do steamers keep steaming in the last
      10–30 min pre-off? Measure move sizes vs the ~2-tick spread +
      commission. This is a known stylised fact — confirm it exists at
      per-minute resolution before spending anything.
- [ ] Baseline signal + flumine simulation mode on Basic data — results are
      screening-grade only (no volume → optimistic fills)
- [ ] Gate: signal survives spread+commission at per-minute resolution →
      buy ONE month of Pro data, re-test with volume-aware fills
- [ ] Later synergy: Phase 3 model-vs-price divergence as an entry signal
      (bet before the market corrects toward the model)

Risks: pre-off trading is the most crowded quant niche on Betfair (BetAngel
crowd + syndicates); per-minute data may be too coarse to see the edge that
exists at tick level. The gate structure caps spend until the signal shows.

## Fallback — only if rpscrape is broken

- [ ] Minimal own scraper: requests/curl_cffi + BS4, results-by-date index,
      resumable via scraped-race-id log, polite rate limit. No Selenium.

## Decisions made

- Temporal split (not per-horse); scaler fit on train only; odds/length/
  current-race ratings excluded from features (leakage).
- `odds` kept as benchmark column only, never a model input.
- og_* pedigree features dropped (leaked, mostly out-of-window); prog_*
  rebuilt as past-only cumulative stats.
- Elo rating added (pre-race value, past-only) — biggest single feature win
  in the CUHK transformer result, and mitigates UK sparse-history problem.
