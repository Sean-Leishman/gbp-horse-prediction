import pandas as pd
import numpy as np
from helper import convertDateToInt, convertStringIntoDate, going_basic_dict, going_dict, dist_dict, type_dict
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
        self.df = self.df.reset_index()
        self.df = self.df.set_index(['jockey_ids', 'horse_ids', 'race_id'])
        self.df['jockey_win_percent'] = self.df.groupby('trainer_ids')['won'].rolling(
            20, closed='left', min_periods=1).mean().reset_index(0, drop=True)

        self.df = self.df.reset_index()
        self.df = self.df.set_index(['trainer_ids', 'horse_ids', 'race_id'])
        self.df['trainer_win_percent'] = self.df.groupby('trainer_ids')['won'].rolling(
            20, closed='left', min_periods=1).mean().reset_index(0, drop=True)

    def compute_pedigree_group(self, group_cols):
        sire_win_percent = self.df.groupby("sire_id")['won'].mean().fillna(
            value=0)._append(pd.Series(0))
        dam_win_percent = self.df.groupby("dam_id")['won'].mean().fillna(
            value=0)._append(pd.Series(0))
        dam_sire_win_percent = self.df.groupby(
            "dam_sire_id")['won'].mean().fillna(value=0)._append(pd.Series(0))

        self.df[['sire_id', 'dam_id', 'dam_sire_id']] = self.df[[
            'sire_id', 'dam_id', 'dam_sire_id']].fillna(value=0)
        self.df['sire_win_percent'] = self.df.apply(
            lambda row: sire_win_percent[row['sire_id']], axis=1)
        self.df['dam_win_percent'] = self.df.apply(
            lambda row: dam_win_percent[row['dam_id']], axis=1)
        self.df['dam_sire_win_percent'] = self.df.apply(
            lambda row: dam_sire_win_percent[row['dam_sire_id']], axis=1)
        # test_df.apply(lambda row: set_sire_win_percent(row),axis=1)

        sire_df = pd.DataFrame(
            data=self.df['sire_id'].unique(), columns=["sire_id"])
        sire_df = pd.DataFrame(
            data=self.df['sire_id'].unique(), columns=["sire_id"])
        sire_df = sire_df.set_index('sire_id')

        self.df = self.df.reset_index()
        self.df = self.df.set_index(["race_id", "horse_ids"])
        new_df = self.df.loc[self.df.index.isin(sire_df.index, level=1)]

        og_win_percent = self.df.loc[self.df.index.isin(sire_df.index, level=1)].groupby(
            level=[1])['won'].mean()
        og_top_speeds = self.df.loc[self.df.index.isin(sire_df.index, level=1)].groupby(
            level=[1])['top_speeds'].mean()
        og_ratings = self.df.loc[self.df.index.isin(sire_df.index, level=1)].groupby(
            level=[1])['ratings'].mean()

        sire_df['num_runners'] = self.df.groupby('sire_id')['won'].count()
        sire_df['won'] = self.df.groupby('sire_id')['won'].sum()
        sire_df['og_win_percent'] = og_win_percent
        sire_df['og_mean_top_speeds'] = og_top_speeds
        sire_df['og_mean_ratings'] = og_ratings
        sire_df['og_flat_win_percent'] = new_df.loc[self.df.race_type == 1].groupby(level=[
            1])['won'].mean()
        sire_df['og_chase_win_percent'] = new_df.loc[self.df.race_type == 2].groupby(level=[
            1])['won'].mean()
        sire_df['og_hurdle_win_percent'] = new_df.loc[self.df.race_type == 0].groupby(level=[
            1])['won'].mean()
        sire_df['og_soft_win_percent'] = new_df.loc[(self.df.going == 13) | (
            self.df.going == 19) | (self.df.going == 14)].groupby(level=[1])['won'].mean()
        sire_df['og_firm_win_percent'] = new_df.loc[(self.df.going == 2) | (
            self.df.going == 1)].groupby(level=[1])['won'].mean()
        sire_df['og_heavy_win_percent'] = new_df.loc[(self.df.going == 8) | (self.df.going == 11) | (
            self.df.going == 12) | (self.df.going == 14)].groupby(level=[1])['won'].mean()
        sire_df['og_good_soft_win_percent'] = new_df.loc[(self.df.going == 6) | (self.df.going == 20) | (
            self.df.going == 7) | (self.df.going == 21) | (self.df.going == 18)].groupby(level=[1])['won'].mean()
        sire_df['og_good_win_percent'] = new_df.loc[(self.df.going == 4) | (
            self.df.going == 16) | (self.df.going == 17)].groupby(level=[1])['won'].mean()
        sire_df['og_good_firm_win_percent'] = new_df.loc[(self.df.going == 5) | (
            self.df.going == 4) | (self.df.going == 2)].groupby(level=[1])['won'].mean()
        sire_df['og_0_6_win_percent'] = new_df.loc[(self.df.dist_categories >= 0) & (
            self.df.dist_categories <= 6)].groupby(level=[1])['won'].mean()
        sire_df['og_7_9_win_percent'] = new_df.loc[(self.df.dist_categories >= 7) & (
            self.df.dist_categories <= 9)].groupby(level=[1])['won'].mean()
        sire_df['og_10_13_win_percent'] = new_df.loc[(self.df.dist_categories >= 10) & (
            self.df.dist_categories <= 13)].groupby(level=[1])['won'].mean()
        sire_df['og_14_20_win_percent'] = new_df.loc[(self.df.dist_categories >= 14) & (
            self.df.dist_categories <= 20)].groupby(level=[1])['won'].mean()
        sire_df['og_21_30_win_percent'] = new_df.loc[(self.df.dist_categories >= 21) & (
            self.df.dist_categories <= 30)].groupby(level=[1])['won'].mean()
        sire_df['og_31_40_win_percent'] = new_df.loc[(self.df.dist_categories >= 31) & (
            self.df.dist_categories <= 40)].groupby(level=[1])['won'].mean()

        sire_df['prog_win_percent'] = self.df.groupby(
            'sire_id')['sire_win_percent'].mean()
        sire_df['prog_mean_top_speeds'] = self.df.groupby('sire_id')[
            'top_speeds'].mean()
        sire_df['prog_mean_ratings'] = self.df.groupby('sire_id')[
            'ratings'].mean()
        sire_df['prog_flat_win_percent'] = self.df.loc[self.df.race_type ==
                                                       1].groupby('sire_id')['won'].mean()
        sire_df['prog_chase_win_percent'] = self.df.loc[self.df.race_type ==
                                                        2].groupby('sire_id')['won'].mean()
        sire_df['prog_hurdle_win_percent'] = self.df.loc[self.df.race_type ==
                                                         0].groupby('sire_id')['won'].mean()
        sire_df['prog_soft_win_percent'] = self.df.loc[(self.df.going == 13) | (
            self.df.going == 19) | (self.df.going == 14)].groupby('sire_id')['won'].mean()
        sire_df['prog_firm_win_percent'] = self.df.loc[(self.df.going == 2) | (
            self.df.going == 1)].groupby('sire_id')['won'].mean()
        sire_df['prog_heavy_win_percent'] = self.df.loc[(self.df.going == 8) | (self.df.going == 11) | (
            self.df.going == 12) | (self.df.going == 14)].groupby('sire_id')['won'].mean()
        sire_df['prog_good_soft_win_percent'] = self.df.loc[(self.df.going == 6) | (self.df.going == 20) | (
            self.df.going == 7) | (self.df.going == 21) | (self.df.going == 18)].groupby('sire_id')['won'].mean()
        sire_df['prog_good_win_percent'] = self.df.loc[(self.df.going == 4) | (
            self.df.going == 16) | (self.df.going == 17)].groupby('sire_id')['won'].mean()
        sire_df['prog_good_firm_win_percent'] = self.df.loc[(self.df.going == 5) | (
            self.df.going == 4) | (self.df.going == 2)].groupby('sire_id')['won'].mean()
        sire_df['prog_0_6_win_percent'] = self.df.loc[(self.df.dist_categories >= 0) & (
            self.df.dist_categories <= 6)].groupby('sire_id')['won'].mean()
        sire_df['prog_7_9_win_percent'] = self.df.loc[(self.df.dist_categories >= 7) & (
            self.df.dist_categories <= 9)].groupby('sire_id')['won'].mean()
        sire_df['prog_10_13_win_percent'] = self.df.loc[(self.df.dist_categories >= 10) & (
            self.df.dist_categories <= 13)].groupby('sire_id')['won'].mean()
        sire_df['prog_14_20_win_percent'] = self.df.loc[(self.df.dist_categories >= 12) & (
            self.df.dist_categories <= 20)].groupby('sire_id')['won'].mean()
        sire_df['prog_21_30_win_percent'] = self.df.loc[(self.df.dist_categories >= 21) & (
            self.df.dist_categories <= 30)].groupby('sire_id')['won'].mean()
        sire_df['prog_31_40_win_percent'] = self.df.loc[(self.df.dist_categories >= 31) & (
            self.df.dist_categories <= 40)].groupby('sire_id')['won'].mean()

        dam_df = pd.DataFrame(
            data=self.df['dam_id'].unique(), columns=["dam_id"])
        dam_df = dam_df.set_index('dam_id')

        new_df = self.df.loc[self.df.index.isin(dam_df.index, level=1)]

        og_win_percent = self.df.loc[self.df.index.isin(dam_df.index, level=1)].groupby(
            level=[1])['won'].mean()
        og_top_speeds = self.df.loc[self.df.index.isin(dam_df.index, level=1)].groupby(
            level=[1])['top_speeds'].mean()
        og_ratings = self.df.loc[self.df.index.isin(dam_df.index, level=1)].groupby(
            level=[1])['ratings'].mean()

        dam_df['num_runners'] = self.df.groupby('dam_id')['won'].count()
        dam_df['won'] = self.df.groupby('dam_id')['won'].sum()
        dam_df['og_win_percent'] = og_win_percent
        dam_df['og_mean_top_speeds'] = og_top_speeds
        dam_df['og_mean_ratings'] = og_ratings
        dam_df['og_flat_win_percent'] = new_df.loc[self.df.race_type == 1].groupby(level=[
            1])['won'].mean()
        dam_df['og_chase_win_percent'] = new_df.loc[self.df.race_type == 2].groupby(level=[
            1])['won'].mean()
        dam_df['og_hurdle_win_percent'] = new_df.loc[self.df.race_type == 0].groupby(level=[
            1])['won'].mean()
        dam_df['og_soft_win_percent'] = new_df.loc[(self.df.going == 13) | (
            self.df.going == 19) | (self.df.going == 14)].groupby(level=[1])['won'].mean()
        dam_df['og_firm_win_percent'] = new_df.loc[(self.df.going == 2) | (
            self.df.going == 1)].groupby(level=[1])['won'].mean()
        dam_df['og_heavy_win_percent'] = new_df.loc[(self.df.going == 8) | (self.df.going == 11) | (
            self.df.going == 12) | (self.df.going == 14)].groupby(level=[1])['won'].mean()
        dam_df['og_good_soft_win_percent'] = new_df.loc[(self.df.going == 6) | (self.df.going == 20) | (
            self.df.going == 7) | (self.df.going == 21) | (self.df.going == 18)].groupby(level=[1])['won'].mean()
        dam_df['og_good_win_percent'] = new_df.loc[(self.df.going == 4) | (
            self.df.going == 16) | (self.df.going == 17)].groupby(level=[1])['won'].mean()
        dam_df['og_good_firm_win_percent'] = new_df.loc[(self.df.going == 5) | (
            self.df.going == 4) | (self.df.going == 2)].groupby(level=[1])['won'].mean()
        dam_df['og_0_6_win_percent'] = new_df.loc[(self.df.dist_categories >= 0) & (
            self.df.dist_categories <= 6)].groupby(level=[1])['won'].mean()
        dam_df['og_7_9_win_percent'] = new_df.loc[(self.df.dist_categories >= 7) & (
            self.df.dist_categories <= 9)].groupby(level=[1])['won'].mean()
        dam_df['og_10_13_win_percent'] = new_df.loc[(self.df.dist_categories >= 10) & (
            self.df.dist_categories <= 13)].groupby(level=[1])['won'].mean()
        dam_df['og_14_20_win_percent'] = new_df.loc[(self.df.dist_categories >= 14) & (
            self.df.dist_categories <= 20)].groupby(level=[1])['won'].mean()
        dam_df['og_21_30_win_percent'] = new_df.loc[(self.df.dist_categories >= 21) & (
            self.df.dist_categories <= 30)].groupby(level=[1])['won'].mean()
        dam_df['og_31_40_win_percent'] = new_df.loc[(self.df.dist_categories >= 31) & (
            self.df.dist_categories <= 40)].groupby(level=[1])['won'].mean()

        dam_df['prog_win_percent'] = self.df.groupby('dam_id')[
            'dam_win_percent'].mean()
        dam_df['prog_mean_top_speeds'] = self.df.groupby('dam_id')[
            'top_speeds'].mean()
        dam_df['prog_mean_ratings'] = self.df.groupby('dam_id')[
            'ratings'].mean()
        dam_df['prog_flat_win_percent'] = self.df.loc[self.df.race_type ==
                                                      1].groupby('dam_id')['won'].mean()
        dam_df['prog_chase_win_percent'] = self.df.loc[self.df.race_type ==
                                                       2].groupby('dam_id')['won'].mean()
        dam_df['prog_hurdle_win_percent'] = self.df.loc[self.df.race_type ==
                                                        0].groupby('dam_id')['won'].mean()
        dam_df['prog_soft_win_percent'] = self.df.loc[(self.df.going == 13) | (
            self.df.going == 19) | (self.df.going == 14)].groupby('dam_id')['won'].mean()
        dam_df['prog_firm_win_percent'] = self.df.loc[(self.df.going == 2) | (
            self.df.going == 1)].groupby('dam_id')['won'].mean()
        dam_df['prog_heavy_win_percent'] = self.df.loc[(self.df.going == 8) | (self.df.going == 11) | (
            self.df.going == 12) | (self.df.going == 14)].groupby('dam_id')['won'].mean()
        dam_df['prog_good_soft_win_percent'] = self.df.loc[(self.df.going == 6) | (self.df.going == 20) | (
            self.df.going == 7) | (self.df.going == 21) | (self.df.going == 18)].groupby('dam_id')['won'].mean()
        dam_df['prog_good_win_percent'] = self.df.loc[(self.df.going == 4) | (
            self.df.going == 16) | (self.df.going == 17)].groupby('dam_id')['won'].mean()
        dam_df['prog_good_firm_win_percent'] = self.df.loc[(self.df.going == 5) | (
            self.df.going == 4) | (self.df.going == 2)].groupby('dam_id')['won'].mean()
        dam_df['prog_0_6_win_percent'] = self.df.loc[(self.df.dist_categories >= 0) & (
            self.df.dist_categories <= 6)].groupby('dam_id')['won'].mean()
        dam_df['prog_7_9_win_percent'] = self.df.loc[(self.df.dist_categories >= 7) & (
            self.df.dist_categories <= 9)].groupby('dam_id')['won'].mean()
        dam_df['prog_10_13_win_percent'] = self.df.loc[(self.df.dist_categories >= 10) & (
            self.df.dist_categories <= 13)].groupby('dam_id')['won'].mean()
        dam_df['prog_14_20_win_percent'] = self.df.loc[(self.df.dist_categories >= 14) & (
            self.df.dist_categories <= 20)].groupby('dam_id')['won'].mean()
        dam_df['prog_21_30_win_percent'] = self.df.loc[(self.df.dist_categories >= 21) & (
            self.df.dist_categories <= 30)].groupby('dam_id')['won'].mean()
        dam_df['prog_31_40_win_percent'] = self.df.loc[(self.df.dist_categories >= 31) & (
            self.df.dist_categories <= 40)].groupby('dam_id')['won'].mean()

        def get_sire_stats(row):
            result = sire_df.loc[row['sire_id']]
            return result["og_"+going_dict[row['going']]], result["prog_"+going_dict[row['going']]], result["og_"+type_dict[row['race_type']]], result["prog_"+type_dict[row['race_type']]], result["og_"+dist_dict[row['dist_categories']]], result["prog_"+dist_dict[row['dist_categories']]]

        self.df['sire_og_going_win_percent'], self.df['sire_prog_going_win_percent'], self.df['sire_og_type_win_percent'], self.df['sire_prog_type_win_percent'], self.df[
            'sire_og_dist_win_percent'], self.df['sire_prog_dist_win_percent'] = zip(*self.df.apply(lambda row: get_sire_stats(row), axis=1))

        self.df['dam_og_going_win_percent'], self.df['dam_prog_going_win_percent'], self.df['dam_og_type_win_percent'], self.df['dam_prog_type_win_percent'], self.df[
            'dam_og_dist_win_percent'], self.df['dam_prog_dist_win_percent'] = zip(*self.df.apply(lambda row: get_sire_stats(row), axis=1))

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
        print("FILLED NAN")
        self.compute_horse_features(['going', 'dist'])
        print("COMPUTED HORSE FEATURES")
        self.compute_features_group()
        print("COMPUTED PEDIGREE GROUP")
        self.compute_pedigree_group(['sire', 'dam', 'dam_sire'])

        self.df.to_csv("fff.csv")


if __name__ == "__main__":
    p = Preprocessor()
    p.preprocess()
