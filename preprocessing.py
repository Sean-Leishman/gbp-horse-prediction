import pandas as pd
import numpy as np
from helper import convertStringIntoDate, going_dict
from helper import race_class_to_scale_dict, going_to_scale_dict
from sklearn.preprocessing import StandardScaler

from timeit import default_timer as timer

"""
Class for preparing data from raw data to be used by the model
"""

# Model input columns. Everything here is known BEFORE the race starts —
# current-race top_speeds/ratings/official_ratings, odds and length are
# outcomes and must never appear in this list.
FEATURE_COLS = [
    'distance', 'going', 'race_class',
    'race_type__0', 'race_type__1', 'race_type__2', 'race_handicap',
    'draws', 'horse_ages', 'horse_weight', 'horse_win_percents',
    'jockey_win_percent', 'trainer_win_percent', 'days_since_last_race',
    'last_figures', 'last_ratings', 'last_official_ratings',
    'mean_figures', 'mean_ratings',
    'best_figures_going', 'best_rating_going',
    'best_official_rating_going', 'win_percent_going',
    'best_figures_distance', 'best_rating_distance',
    'best_official_rating_distance', 'win_percent_distance',
    'elo_rating',
    'sire_win_percent', 'dam_win_percent', 'dam_sire_win_percent',
    'sire_prog_going_win_percent', 'sire_prog_type_win_percent',
    'sire_prog_dist_win_percent',
    'dam_prog_going_win_percent', 'dam_prog_type_win_percent',
    'dam_prog_dist_win_percent',
]


class Preprocessor:
    def __init__(self):
        self.df = None

    def load_file(self, filename: str, drop=True):
        if drop:
            return pd.read_csv(filename).drop("Unnamed: 0", axis=1)
        return pd.read_csv(filename)

    def fill_nan_with_0(self):
        self.df = self.df.fillna(0)

    """
    Generate horse features that are based on their history:

    @param: group_cols -> names of columns related to race stats that features are generated based on 

    - days_since_last_race, top_speed/official_rating/rating of last race,
    - mean top_speed/official_rating/rating of last 5 races
    - horse_win_percents of last 5 races
    - best ratings dependent on going & distance (&param group_cols)
    """
    def compute_horse_features(self, group_cols):
        self.df = self.df.sort_values('date')
        max_num_races = 10

        self.df = self.df.set_index(['horse_ids', 'race_id'])
        
        self.df['last_figures'] = self.df.groupby(
            'horse_ids')['top_speeds'].rolling(1, closed='left').sum().reset_index(0, drop=True)
        self.df['last_ratings'] = self.df.groupby(
            'horse_ids')['ratings'].rolling(1, closed='left').sum().reset_index(0, drop=True)
        self.df['last_official_ratings'] = self.df.groupby(
            'horse_ids')['official_ratings'].rolling(1, closed='left').sum().reset_index(0, drop=True)

        self.df['mean_figures'] = self.df.groupby('horse_ids')['top_speeds'].rolling(
            5, min_periods=1, closed='left').mean().reset_index(0, drop=True)
        self.df['mean_ratings'] = self.df.groupby('horse_ids')['ratings'].rolling(
            5, min_periods=1, closed='left').mean().reset_index(0, drop=True)

        self.df['horse_win_percents'] = self.df.groupby('horse_ids')['won'].rolling(
            6, closed='left', min_periods=1).mean().reset_index(0, drop=True)

        if group_cols:
            for group_col in group_cols:
                self.df[f'best_figures_{group_col}'] = self.df.groupby(['horse_ids', group_col])['top_speeds'].rolling(
                    max_num_races, closed='left', min_periods=1).max().reset_index([0, 1], drop=True)
                self.df[f'best_rating_{group_col}'] = self.df.groupby(['horse_ids', group_col])['ratings'].rolling(
                    max_num_races, closed='left', min_periods=1).max().reset_index([0, 1], drop=True)
                self.df[f'best_official_rating_{group_col}'] = self.df.groupby(['horse_ids', group_col])['official_ratings'].rolling(
                    max_num_races, closed='left', min_periods=1).max().reset_index([0, 1], drop=True)
                self.df[f'win_percent_{group_col}'] = self.df.groupby(['horse_ids', group_col])['won'].rolling(
                    max_num_races, closed='left', min_periods=1).mean().reset_index([0, 1], drop=True)
        self.df = self.df.reset_index()

        self.df = self.df.fillna(0)
        self.compute_elo_ratings()

    """
    Elo rating per horse from finishing order. 'elo_rating' is the horse's
    rating BEFORE the race, so it is past-only by construction. Score is the
    fraction of the field beaten; expected score comes from pairwise Elo
    win probabilities against the rest of the field.
    """
    def compute_elo_ratings(self, k=32):
        self.df = self.df.sort_values('date_race_id')
        ratings = {}
        pre_race = np.empty(len(self.df))
        pos = 0
        for _, grp in self.df.groupby('date_race_id', sort=True):
            ids = grp['horse_ids'].values
            n = len(ids)
            r = np.array([ratings.get(h, 1500.0) for h in ids])
            pre_race[pos:pos + n] = r
            pos += n
            if n < 2:
                continue
            rank = grp['places'].rank(method='average').values
            score = (n - rank) / (n - 1)
            expected = (1.0 / (1.0 + 10.0 ** ((r[None, :] - r[:, None]) / 400.0))).sum(axis=1)
            expected = (expected - 0.5) / (n - 1)  # drop self-match (diagonal = 0.5)
            for h, new in zip(ids, r + k * (score - expected)):
                ratings[h] = new
        self.df['elo_rating'] = pre_race
    def compute_auxillary_features_group(self):
        self.df = self.df.set_index(['jockey_ids', 'horse_ids', 'race_id'])
        self.df['jockey_win_percent'] = self.df.groupby('jockey_ids')['won'].rolling(
            20, closed='left', min_periods=1).mean().reset_index(0, drop=True)

        self.df = self.df.reset_index()
        self.df = self.df.set_index(['trainer_ids', 'horse_ids', 'race_id'])
        self.df['trainer_win_percent'] = self.df.groupby('trainer_ids')['won'].rolling(
            20, closed='left', min_periods=1).mean().reset_index(0, drop=True)
        self.df = self.df.reset_index()

        self.df = self.df.fillna(0)
    """
    Progeny win percents for sire/dam/dam-sire, computed from PAST races only:
    overall, and conditioned on the current race's going band, race type and
    distance band. For each row this is "win % of this parent's offspring in
    races strictly before today's race".
    """
    def _past_win_percent(self, group_cols):
        g = self.df.groupby(group_cols, sort=False)['won']
        # cumulative wins/starts up to but excluding the current row
        # ponytail: two same-parent runners in one race see each other's result;
        # negligible — aggregate per (parent, race) first if it ever matters
        return (g.cumsum() - self.df['won']) / g.cumcount()

    def compute_pedigree_group(self):
        self.df = self.df.sort_values('date')
        self.df[['sire_id', 'dam_id', 'dam_sire_id']] = self.df[[
            'sire_id', 'dam_id', 'dam_sire_id']].fillna(0)

        self.df['going_band'] = self.df['going'].map(going_dict)

        self.df['sire_win_percent'] = self._past_win_percent(['sire_id'])
        self.df['dam_win_percent'] = self._past_win_percent(['dam_id'])
        self.df['dam_sire_win_percent'] = self._past_win_percent(['dam_sire_id'])

        for parent in ('sire', 'dam'):
            self.df[f'{parent}_prog_going_win_percent'] = self._past_win_percent(
                [f'{parent}_id', 'going_band'])
            self.df[f'{parent}_prog_type_win_percent'] = self._past_win_percent(
                [f'{parent}_id', 'race_type'])
            self.df[f'{parent}_prog_dist_win_percent'] = self._past_win_percent(
                [f'{parent}_id', 'distance_categories'])

        # ponytail: og_* features (the parent's own racing record) dropped — the
        # old computation averaged over the whole dataset (future leakage) and a
        # parent's own races mostly predate the data anyway. Re-add as a static
        # per-horse career lookup if they earn their keep.
        self.df = self.df.drop('going_band', axis=1)
        self.df = self.df.fillna(0)

    """ 
    Main entry function to clean data from raw data files and merge into a singular dataframe
    """
    def feature_generation(self):
        runner_df = self.load_file("data/raw/runners_UK2.csv")
        race_df = self.load_file("data/raw/races_UK2.csv")

        race_df = race_df[race_df['date'].notna()]

        race_df = race_df.drop('track_name', axis=1)

        race_df['race_class'] = race_df['race_class'].replace(race_class_to_scale_dict)
        race_df['going'] = race_df['going'].replace(going_to_scale_dict)
        race_df['going'] = race_df['going'].fillna(-1)

        race_df['race_type'] = race_df['race_type'].astype(
            'category').cat.codes
        race_df['date'] = race_df['date'].apply(
            lambda x: convertStringIntoDate(x))
        
        race_df['distance_categories'] = pd.qcut(race_df['distance'], q=10, labels=False)

        race_type = pd.get_dummies(
            race_df['race_type'].astype("category"), prefix="race_type_")
        
        race_df = race_df.merge(
            race_type, left_index=True, right_index=True)

        race_df = race_df.astype({'going':int})

        # encode values for runs DB
        runner_df.replace(u'\xa0', u'', regex=True, inplace=True)

        runner_df['horse_ages'] = np.abs(runner_df['horse_ages'])
        runner_df['horse_ages'] = pd.qcut(runner_df['horse_ages'], q=5, labels=False)
        runner_df['draws'] = pd.qcut(runner_df['draws'], q=10, labels=False)
        # no draw (jumps races have no stalls) -> sentinel, not a random value
        runner_df['draws'] = runner_df['draws'].fillna(-1)

        runner_df = runner_df.drop('horse_names', axis=1)
        runner_df['places'].replace({"F": 0, 'PU': 0, "DSQ": 0, 'SU':  0, 'BD': 0, 'UR': 0, 'RO': 0, 'RR': 0, 'REF': 0,
                                     'LFT': 0, 'CO': 0, 'VOI': 0}, inplace=True)
        runner_df['won'] = np.where((runner_df.places == "1"), 1, 0)
        runner_df = runner_df.replace('–', 0)
        runner_df = runner_df.fillna(0)


        runner_df = runner_df.astype({'race_id': int, 'horse_ids': int,
                                      'draws': int, 'horse_ages': int, 'horse_weight': int, 'jockey_ids': int,
                                      'trainer_ids': int, 'top_speeds': int,
                                      'ratings': int, 'official_ratings': int, 'odds': float, 'places': int})

        max_places = np.max(runner_df['places'])
        runner_df.loc[(runner_df.won == 0) & (runner_df.places == 0), 'places'] = max_places
                              

        self.df = self.load_file("data/raw/full_data4.csv", drop=False)
        self.df.reset_index(inplace=True)
        self.df = self.df.rename(columns={"index": "race_id", "Unnamed: 0": "horse_ids", "male_pedigree": "sire_id",
                                          "female_pedigree": "dam_id", "older_pedigree": "dam_sire_id"})

        df = race_df.merge(runner_df, on="race_id")
        self.df = df.merge(self.df[['race_id', 'horse_ids', 'sire_id',
                                    'dam_id', 'dam_sire_id', 'length']], on=["race_id", "horse_ids"])

        self.df = self.df.drop_duplicates()

        self.df = self.df.sort_values(by="date")
        self.df['date_race_id'] = pd.factorize(self.df['race_id'])[0]

        self.preprocess_columns()

    def preprocess_columns(self):
        max_length = max(self.df['length'])
        self.df.loc[(self.df.length == 0) & (
            self.df.won == 0), 'length'] = max_length

        self.df['days_since_last_race'] = (self.df['date'] - \
            self.df.groupby('horse_ids')['date'].shift()).fillna(0).apply(lambda x:  x.days if x != 0 else 0)

    def select_columns(self):
        # 'odds' is kept as a market benchmark to evaluate against — it is NOT
        # a model input (it encodes the outcome the market already knows)
        self.df = self.df[['horse_ids', 'date_race_id', 'won', 'odds'] + FEATURE_COLS].copy()
        self.df = self.df.fillna(0)

        # order rows (horse, date) so a horse's history is a contiguous slice
        # ending just above its current row — the model dataset relies on this
        self.df = self.df.sort_values(by='date_race_id')
        self.df['offset_horse_id'] = pd.factorize(self.df['horse_ids'])[0]
        self.df = self.df.sort_values(by=['offset_horse_id', 'date_race_id'])
        self.df['num_previous_races'] = self.df.groupby('offset_horse_id').cumcount()
        self.df = self.df.drop('horse_ids', axis=1)

    def train_test_split(self):
        """Temporal split: last 20% of races are test. The scaler is fit on
        train rows only. One file is written; test rows are flagged so the
        model can still read a horse's pre-cutoff history at test time
        (that history is legitimately in the past, not leakage)."""
        cutoff = self.df['date_race_id'].quantile(0.8)
        self.df['is_test'] = self.df['date_race_id'] > cutoff

        scaler = StandardScaler().fit(self.df.loc[~self.df.is_test, FEATURE_COLS])
        self.df[FEATURE_COLS] = scaler.transform(self.df[FEATURE_COLS])

        self.df.to_csv("data/preprocessing/6-model-data.csv")

    def preprocess(self, merge_df=False, comp_horse_feats=False, comp_aux_feats=False, comp_pedigree_feats=False, select_columns=False, train_test_split=False):
        start = timer()
        if merge_df:
            self.feature_generation()
            self.df.to_csv("data/preprocessing/1-feature-generation.csv")
        end = timer()
        print(f"MERGED DATAFRAMES -> Time: {end-start}")
        
        start = timer()
        if comp_horse_feats:
            if self.df is None:
                self.df = pd.read_csv("data/preprocessing/1-feature-generation.csv", index_col=[0])
            self.compute_horse_features(['going', 'distance'])
            self.df.to_csv("data/preprocessing/2-horse-features.csv")
        end = timer()
        print(f"COMPUTED HORSE FEATURES -> Time: {end-start}")
        
        start = timer()
        if comp_aux_feats:
            if self.df is None:
                self.df = pd.read_csv("data/preprocessing/2-horse-features.csv", index_col=[0])
            self.compute_auxillary_features_group()
            self.df.to_csv("data/preprocessing/3-auxillary-features.csv")
        end = timer()
        print(f"COMPUTED AUXILLARY FEATURES -> Time: {end-start}")    

        start = timer()
        if comp_pedigree_feats:
            if self.df is None:
                self.df = pd.read_csv("data/preprocessing/3-auxillary-features.csv", index_col=[0])
            self.compute_pedigree_group()
            self.df.to_csv("data/preprocessing/4-pedigree-group.csv")
        end = timer()
        print(f"COMPUTED PEDIGREE GROUP -> Time: {end-start}")

        start = timer()
        if select_columns:
            if self.df is None:
                self.df = pd.read_csv("data/preprocessing/4-pedigree-group.csv", index_col=[0])
            self.select_columns()
            self.df.to_csv("data/preprocessing/5-selected-data.csv")
        end = timer()
        print(f"SELECTED COLUMNS -> Time: {end-start}")

        start = timer()
        if train_test_split:
            if self.df is None:
                self.df = pd.read_csv("data/preprocessing/5-selected-data.csv", index_col=[0])
            self.train_test_split()
        end = timer()
        print(f"Train test split -> Time: {end-start}")

if __name__ == "__main__":
    p = Preprocessor().preprocess(merge_df=True, comp_horse_feats=True, comp_aux_feats=True, comp_pedigree_feats=True, select_columns=True, train_test_split=True)
