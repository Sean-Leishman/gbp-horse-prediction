"""rpscrape -> model pipeline. Reads rpscrape year CSVs, builds the merged
encoded frame the Preprocessor stages expect (same shape test_pipeline.py
synthesises), runs the full feature pipeline, writes 6-model-data.csv.

Usage: python data_import.py [rpscrape_region_dir ...]
       default: ~/Projects/rpscrape/data/region/{gb,ire}
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from helper import going_to_scale_dict
from preprocessing import Preprocessor

RPSCRAPE_REGION = Path.home() / 'Projects/rpscrape/data/region'

# finish-position codes that mean "did not finish/place" (legacy list + rpscrape's)
NON_FINISH = {'F', 'PU', 'DSQ', 'SU', 'BD', 'UR', 'RO', 'RR', 'REF',
              'LFT', 'CO', 'VOI', 'DNF', '0'}

TYPE_CODE = {'Hurdle': 0, 'Flat': 1, 'Chase': 2}  # matches helper.type_dict


def to_num(s):
    return pd.to_numeric(s.replace({'–': None, '-': None, '': None}), errors='coerce')


def load_rpscrape(dirs):
    files = sorted(f for d in dirs for f in Path(d).rglob('*.csv'))
    if not files:
        sys.exit(f'no csv files under {dirs}')
    df = pd.concat((pd.read_csv(f, dtype=str) for f in files), ignore_index=True)
    print(f'{len(files)} files, {len(df)} rows')
    return df



def course_draw_bias(df, course, train_rows, min_cell=200):
    """Historical win-rate edge of this draw position AT THIS COURSE, relative
    to the course's own base rate. Estimated on the training window only —
    the same discipline the scaler already follows — so it cannot leak the
    test period. Cells with too little history shrink to 0 (no opinion).
    """
    bucket = pd.qcut(df['draw_pct'], 3, labels=False, duplicates='drop')
    tr = pd.DataFrame({'course': course[train_rows], 'b': bucket[train_rows],
                       'won': df['won'][train_rows]})
    cell = tr.groupby(['course', 'b']).won.agg(['mean', 'size'])
    base = tr.groupby('course').won.mean()
    bias = (cell['mean'] - base.reindex(cell.index.get_level_values('course')).values)
    bias = bias.where(cell['size'] >= min_cell, 0.0).to_dict()
    return pd.Series(list(zip(course, bucket)), index=df.index).map(bias).fillna(0.0)


def build_frame(raw):
    raw = raw[raw['type'].isin(TYPE_CODE)].copy()  # drops NH Flat etc (~2%)

    df = pd.DataFrame()
    df['race_id'] = to_num(raw['race_id']).astype('Int64')
    df['date'] = pd.to_datetime(raw['date'])
    df['horse_ids'] = to_num(raw['horse_id']).astype('Int64')
    df['jockey_ids'] = to_num(raw['jockey_id']).fillna(0).astype(int)
    df['trainer_ids'] = to_num(raw['trainer_id']).fillna(0).astype(int)
    df['sire_id'] = to_num(raw['sire_id']).fillna(0).astype(int)
    df['dam_id'] = to_num(raw['dam_id']).fillna(0).astype(int)
    df['dam_sire_id'] = to_num(raw['damsire_id']).fillna(0).astype(int)

    # going: compound forms like "Good (Good To Soft In Places)" -> main part
    going = raw['going'].fillna('').str.split(' (', regex=False).str[0].str.strip().str.title()
    df['going'] = going.map({k.title(): v for k, v in going_to_scale_dict.items()}).fillna(-1).astype(int)
    unmapped = going[~going.isin([k.title() for k in going_to_scale_dict])].value_counts()
    if len(unmapped):
        print('unmapped goings ->-1:', dict(unmapped.head(10)))

    df['distance'] = to_num(raw['dist_m']).fillna(0).astype(int)
    df['distance_categories'] = pd.qcut(df['distance'], q=10, labels=False, duplicates='drop')

    # class: "Class 3" -> 5 (legacy scale 8-n); Group/Listed pattern -> top class
    cls = raw['class'].fillna('').str.extract(r'Class (\d)')[0]
    pattern_cls = pd.Series(np.where(raw['pattern'].fillna('') != '', 7, 0), index=raw.index)
    df['race_class'] = (8 - to_num(cls)).fillna(pattern_cls).astype(int)

    df['race_type'] = raw['type'].map(TYPE_CODE).astype(int)
    for t in (0, 1, 2):
        df[f'race_type__{t}'] = (df['race_type'] == t).astype(int)

    df['race_handicap'] = raw['race_name'].fillna('').str.lower().str.contains(
        'handicap|nursery|h\'cap').astype(int)

    # Draw. A global decile of stall NUMBER encodes field size, not draw bias:
    # pooled win rate by decile falls 0.121 -> 0.064 purely because a high
    # stall only exists in a big field. Real draw bias is per course and
    # reverses sign (Chester +6.5pp for low draws, Lingfield -3.1pp), so
    # pooling averages it to nothing. See diag_draw.py.
    draws = to_num(raw['draw'])
    df['draw_pct'] = draws.groupby(raw['race_id']).rank(pct=True).fillna(0.5)
    df['headgear'] = (raw['hg'].fillna('') != '').astype(int)
    # rpscrape suffixes a '1' for first time in that headgear (v1, b1, ...) —
    # a first-time-blinkers signal the market prices and we had nothing for.
    df['first_time_headgear'] = raw['hg'].fillna('').str.contains('1').astype(int)

    df['horse_ages'] = pd.qcut(to_num(raw['age']).abs(), q=5, labels=False, duplicates='drop')
    df['horse_ages'] = df['horse_ages'].fillna(0).astype(int)
    df['horse_weight'] = to_num(raw['lbs']).fillna(0).astype(int)

    # Ratings: 0 is NOT "unknown" — it reads as "worse than any horse ever
    # rated" (median RPR is 62), and rpr/ts/or are missing on 5.5/12.4/19.8%
    # of runners. Impute the median and flag it, so the model can tell an
    # unrated horse from a bad one. Fixed here rather than downstream because
    # the Preprocessor derives last_/mean_/best_* features from these columns,
    # so a zero here poisons the whole rating chain.
    train_rows = df['date'] <= df['date'].quantile(0.8)   # same boundary the
    for src, dst in (('rpr', 'ratings'), ('ts', 'top_speeds'),             # split uses
                     ('or', 'official_ratings')):
        df[dst] = to_num(raw[src]).fillna(
            to_num(raw[src])[train_rows].median()).astype(int)

    # DO NOT add rpr/ts missingness as a feature. Racing Post withholds an RPR
    # from horses beaten a long way, so `rpr is null` is read off the RESULT:
    # 22,276 such runners 2009-15 contain 2 winners (0.01% vs a 10.7% base),
    # and ts is nearly as bad at 1.10%. Adding them scored a spectacular
    # +0.051 nats on the second-stage gate that was entirely this leak.
    # `or` is different and safe: an official mark is assigned before the race,
    # and missing-or runners win at 10.02% vs 10.86% — no outcome information.
    df['official_ratings_missing'] = to_num(raw['or']).isna().astype(int)

    # benchmark odds: BSP (margin-free) where matched, bookmaker decimal otherwise
    bsp, dec = to_num(raw['bsp']), to_num(raw['dec'])
    print(f'bsp missing: {bsp.isna().mean()*100:.1f}% (filled from dec)')
    df['odds'] = bsp.fillna(dec).fillna(0).astype(float)
    # Betfair MORNINGWAP: a price known BEFORE the bet, for ROI decisions (D-005).
    # BSP settles after the bet, so deciding on it is a lookahead. 0 = untraded.
    df['morning_wap'] = to_num(raw['morning_wap']).fillna(0).astype(float)

    pos = raw['pos'].fillna('0').str.strip()
    pos = pos.where(~pos.isin(NON_FINISH), '0')
    df['places'] = to_num(pos).fillna(0).astype(int)
    df['won'] = (df['places'] == 1).astype(int)
    max_places = df['places'].max()
    df.loc[(df.won == 0) & (df.places == 0), 'places'] = max_places

    df['length'] = to_num(raw['ovr_btn']).fillna(0).astype(float)
    df['course_draw_bias'] = course_draw_bias(df, raw['course'], train_rows)

    df = df.dropna(subset=['race_id', 'horse_ids'])
    df = df.astype({'race_id': int, 'horse_ids': int})
    df = df.drop_duplicates(subset=['race_id', 'horse_ids'])
    df = df.sort_values('date', kind='stable')
    df['date_race_id'] = pd.factorize(df['race_id'])[0]
    return df


def main(dirs):
    df = build_frame(load_rpscrape(dirs))
    print(f'{len(df)} runners, {df.race_id.nunique()} races, '
          f'{df.date.min().date()} -> {df.date.max().date()}, '
          f'win rate {df.won.mean()*100:.1f}%')

    p = Preprocessor()
    p.df = df
    p.preprocess_columns()
    p.compute_horse_features(['going', 'distance'])
    p.compute_auxillary_features_group()
    p.compute_pedigree_group()
    p.select_columns()
    Path('data/preprocessing').mkdir(parents=True, exist_ok=True)
    p.train_test_split()
    print('written data/preprocessing/6-model-data.csv')


if __name__ == '__main__':
    # NOT region/gb wholesale: region/<r>/all holds day-mode smoke files
    # (2015-06-10, 2024-06-12, 2026-07-10) — isolated days with no surrounding
    # history that land at the end of the split and inside any holdout window.
    main(sys.argv[1:] or [RPSCRAPE_REGION / r / t
                          for r in ('gb', 'ire') for t in ('flat', 'jumps')])
