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

    draws = to_num(raw['draw'])
    df['draws'] = pd.qcut(draws, q=10, labels=False, duplicates='drop')
    df['draws'] = df['draws'].fillna(-1).astype(int)  # jumps: no stalls

    df['horse_ages'] = pd.qcut(to_num(raw['age']).abs(), q=5, labels=False, duplicates='drop')
    df['horse_ages'] = df['horse_ages'].fillna(0).astype(int)
    df['horse_weight'] = to_num(raw['lbs']).fillna(0).astype(int)

    df['top_speeds'] = to_num(raw['ts']).fillna(0).astype(int)
    df['ratings'] = to_num(raw['rpr']).fillna(0).astype(int)
    df['official_ratings'] = to_num(raw['or']).fillna(0).astype(int)

    # benchmark odds: BSP (margin-free) where matched, bookmaker decimal otherwise
    bsp, dec = to_num(raw['bsp']), to_num(raw['dec'])
    print(f'bsp missing: {bsp.isna().mean()*100:.1f}% (filled from dec)')
    df['odds'] = bsp.fillna(dec).fillna(0).astype(float)

    pos = raw['pos'].fillna('0').str.strip()
    pos = pos.where(~pos.isin(NON_FINISH), '0')
    df['places'] = to_num(pos).fillna(0).astype(int)
    df['won'] = (df['places'] == 1).astype(int)
    max_places = df['places'].max()
    df.loc[(df.won == 0) & (df.places == 0), 'places'] = max_places

    df['length'] = to_num(raw['ovr_btn']).fillna(0).astype(float)

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
    main(sys.argv[1:] or [RPSCRAPE_REGION / 'gb', RPSCRAPE_REGION / 'ire'])
