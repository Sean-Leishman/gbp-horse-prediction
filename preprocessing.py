import pandas as pd
import numpy as np
from helper import convertDateToInt, convertStringIntoDate
from sklearn.linear_model import LinearRegression

"""
Class for preparing data from raw data to be used by the model
"""


class Preprocessor:
    def __init__(self):
        self.filename = ''
        self.df = None

    def load_file(self, filename: str, drop=True):
        if drop:
            return pd.read_csv(filename).drop("Unnamed: 0", axis=1)
        return pd.read_csv(filename)

    def fill_nan_with_0(self):
        self.df = self.df.fillna(0)

    def fill_nan_with_lr(self):
        train_rating = self.df.loc[self.df.feature_name != 0]
        predict_of = self.df.loc[(self.df.ratings != 0) & (
            self.df.top_speeds != 0) & (self.df.official_ratings == 0)]
        predict_speed_on_rating = self.df.loc[(
            self.df.ratings == 0) & (self.df.top_speeds != 0)]

        train_data = self.df.loc[(self.df.ratings != 0) & (
            self.df.top_speeds != 0) & (self.df.official_ratings != 0)]

        # top_speeds nan fill
        train_rating = self.df.loc[self.df.top_speeds != 0]

        predict_rating_on_speed = self.df.loc[(
            self.df.ratings != 0) & (self.df.top_speeds == 0)]
        predict_speed_on_rating = self.df.loc[(
            self.df.ratings == 0) & (self.df.top_speeds != 0)]

        train_data = self.df.loc[(self.df.ratings != 0)
                                 & (self.df.top_speeds != 0)]

        model = LinearRegression()
        model.fit(train_data[['top_speeds', 'ratings']].values,
                  train_data['official_ratings'].values.reshape(-1, 1))
        pred = model.predict(predict_of[['top_speeds', 'ratings']].values)
        self.df.loc[predict_of.index, 'official_ratings'] = pred.flatten()

        pred = model.predict(
            predict_rating_on_speed['ratings'].values.reshape(-1, 1))
        self.df.loc[predict_rating_on_speed.index,
                    'top_speeds'] = pred.flatten()

        model = LinearRegression()
        model.fit(train_data['top_speeds'].values.reshape(-1, 1),
                  train_data['ratings'].values.reshape(-1, 1))
        pred = model.predict(
            predict_speed_on_rating['top_speeds'].values.reshape(-1, 1))
        self.df.loc[predict_speed_on_rating.index, 'ratings'] = pred.flatten()

    def compute_horse_features(self, group_cols):
        self.df = self.df.sort_values('date')
        max_num_races = 10

        self.df = self.df.set_index(['horse_ids', 'race_id'])

        self.df['days_since_last_race'] = self.df['date'] - \
            self.df.groupby('horse_ids')['date'].shift()
        print(self.df)
        print(self.df.groupby(
            'horse_ids')['top_speeds'].rolling(1, closed='left').sum().reset_index(0, drop=True))
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

        self.df['difference_in_ratings'] = self.df['last_ratings'] - \
            self.df['last_ratings']
        self.df['horse_win_percents'] = self.df.groupby('horse_ids')['won'].rolling(
            6, closed='left', min_periods=1).mean().reset_index(0, drop=True)

        if group_cols:
            for group_col in group_cols:
                if group_col == 'dist':
                    self.df['dist_categories'] = (
                        self.df['distance']/201.168).round()
                elif group_col == "going":
                    self.df['going_cat'] = self.df['going'].replace(
                        going_basic_dict)

                self.df[f'best_figures_{group_col}'] = self.df.groupby(['horse_ids', 'going'])['top_speeds'].rolling(
                    max_num_races, closed='left', min_periods=1).max().reset_index([0, 1], drop=True)
                self.df[f'best_rating_{group_col}'] = self.df.groupby(['horse_ids', 'going'])['ratings'].rolling(
                    max_num_races, closed='left', min_periods=1).max().reset_index([0, 1], drop=True)
                self.df[f'best_official_rating_{group_col}'] = self.df.groupby(['horse_ids', 'going'])['official_ratings'].rolling(
                    max_num_races, closed='left', min_periods=1).max().reset_index([0, 1], drop=True)
                self.df[f'win_percent_{group_col}'] = self.df.groupby(['horse_ids', 'going'])['won'].rolling(
                    max_num_races, closed='left', min_periods=1).mean().reset_index([0, 1], drop=True)

    def compute_features_group(self):
        jockey_df = self.df.set_index(['jockey_ids', 'horse_ids', 'race_id'])
        trainer_df = self.df.set_index(['trainer_ids', 'horse_ids', 'race_id'])
        self.df['jockey_win_percent'] = jockey_df.groupby('trainer_ids')['won'].rolling(
            20, closed='left', min_periods=1).mean().reset_index(0, drop=True)

        self.df['trainer_win_percent'] = trainer_df.groupby('trainer_ids')['won'].rolling(
            20, closed='left', min_periods=1).mean().reset_index(0, drop=True)

    def compute_pedigree_group(self, group_cols):
        pass

    def preprocess(self):
        runner_df = self.load_file("data/raw/runners_UK2.csv")
        race_df = self.load_file("data/raw/races_UK2.csv")

        race_df = race_df[race_df['date'].notna()]

        # track_id should be track_name
        race_df.rename(columns={0: 'race_id'})
        race_df = race_df.drop('track_name', axis=1)
        race_df['race_class'] = race_df['race_class'].astype(
            'category').cat.codes
        race_df['going'] = race_df['going'].astype('category').cat.codes
        race_df['race_age'] = race_df['race_age'].astype('category').cat.codes
        race_df['race_type'] = race_df['race_type'].astype(
            'category').cat.codes
        race_df['date'] = race_df['date'].apply(
            lambda x: convertStringIntoDate(x))

        # encode values for runs DB
        runner_df.replace(u'\xa0', u'', regex=True, inplace=True)
        runner_df = runner_df.drop('horse_names', axis=1)
        runner_df['places'].replace({"F": 0, 'PU': 0, "DSQ": 0, 'SU': 0, 'BD': 0, 'UR': 0, 'RO': 0, 'RR': 0, 'REF': 0,
                                     'LFT': 0, 'CO': 0, 'VOI': 0}, inplace=True)
        """runner_df['places'] = np.where((runner_df.places == "1" or runner_df.places == "2"
                                    or runner_df.places == "3"), 1, 0)"""
        runner_df['won'] = np.where((runner_df.places == "1"), 1, 0)
        runner_df = runner_df.replace('–', 0)
        runner_df = runner_df.fillna(0)
        # runner_df['true_id'] = runner_df['race_id'] + runner_df['horse_ids']
        runner_df = runner_df.astype({'race_id': int, 'horse_ids': int,
                                      'draws': int, 'horse_ages': int, 'horse_weight': int, 'jockey_ids': int,
                                      'trainer_ids': int, 'top_speeds': int,
                                      'ratings': int, 'official_ratings': int, 'odds': float, 'places': int})
        runner_df['horse_ages'] = runner_df['horse_ages'].astype(
            'category').cat.codes
        runner_df['horse_weight'] = runner_df['horse_weight']

        self.df = self.load_file("data/raw/full_data4.csv", drop=False)
        self.df.reset_index(inplace=True)
        self.df = self.df.rename(columns={"index": "race_id", "Unnamed: 0": "horse_ids", "male_pedigree": "sire_id",
                                          "female_pedigree": "dam_id", "older_pedigree": "dam_sire_id"})

        df = race_df.merge(runner_df, on="race_id")
        self.df = df.merge(self.df[['race_id', 'horse_ids', 'sire_id',
                                    'dam_id', 'dam_sire_id']], on=["race_id", "horse_ids"])

        self.df = self.df.drop_duplicates()

        self.fill_nan_with_0()
        self.compute_horse_features(['going', 'dist'])
        self.compute_features_group()
        self.compute_pedigree_group(['sire', 'dam', 'dam_sire'])


if __name__ == "__main__":
    p = Preprocessor()
    p.preprocess()
