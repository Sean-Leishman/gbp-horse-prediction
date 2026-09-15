"""Same-day leak check (D-004). Features aggregated over OTHER rows' results
must be identical for every row sharing (key, date): anything else means a
same-day -- possibly same-race -- result leaked in. Run after any feature change.

    python leak_check.py        # ~3 min
"""
from data_import import load_rpscrape, build_frame, RPSCRAPE_REGION
from preprocessing import Preprocessor

p = Preprocessor()
p.df = build_frame(load_rpscrape([RPSCRAPE_REGION / 'gb', RPSCRAPE_REGION / 'ire']))
p.preprocess_columns()
p.compute_horse_features(['going', 'distance'])
p.compute_auxillary_features_group()
p.compute_pedigree_group()

bad = []
for key, feat in (('trainer_ids', 'trainer_win_percent'), ('jockey_ids', 'jockey_win_percent'),
                  ('sire_id', 'sire_win_percent'), ('dam_id', 'dam_win_percent'),
                  ('dam_sire_id', 'dam_sire_win_percent')):
    n = int((p.df.groupby([key, 'date'])[feat].nunique() > 1).sum())
    print(f'{feat:22s} (key, day) groups with differing values: {n}')
    bad += [feat] * (n > 0)
assert not bad, f'same-day leak in {bad}'
print('ok')
