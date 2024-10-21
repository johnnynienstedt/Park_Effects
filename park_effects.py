#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:19:26 2024

@author: johnnynienstedt
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import floor, log10
from IPython import get_ipython
from itertools import combinations
from statistics import NormalDist
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



'''
###############################################################################
###################### Data Prep and Function Definition ######################
###############################################################################
'''



# function to retrieve pitch-by-pitch data (2021-2024)
def get_data():
    # import pitch data from 2021-2024
    all_pitch_data = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Pitching Analysis/Shape+/all_pitch_data.csv', index_col=False)
    
    drop_cols = ['Unnamed: 0', 'level_0', 'index']
    necessary_cols = ['release_speed', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0', 'ax',
                      'ay', 'az', 'release_pos_x', 'release_pos_y', 'release_pos_z', 
                      'release_extension']
    
    clean_pitch_data = all_pitch_data.copy().drop(columns = drop_cols)
    clean_pitch_data = clean_pitch_data.dropna(subset = necessary_cols)
    
    # select pitchers with at least 100 pitches thrown
    pitch_data = clean_pitch_data.groupby('pitcher').filter(lambda x: len(x) >= 100)
    
    # flip axis for RHP so that +HB = arm side, -HB = glove side
    mirror_cols = ['release_pos_x', 'plate_x', 'pfx_x', 'vx0', 'ax']
    pitch_data.loc[pitch_data['p_throws'] == 'R', mirror_cols] = -pitch_data.loc[pitch_data['p_throws'] == 'R', mirror_cols]
    
    #
    # Add secondary metrics
    #
    
    # get relevant primary parameters
    vx0 = pitch_data['vx0']
    vy0 = pitch_data['vy0']
    vz0 = pitch_data['vz0']
    ax = pitch_data['ax']
    ay = pitch_data['ay']
    az = pitch_data['az']
    rx = pitch_data['release_pos_x']
    ry = pitch_data['release_pos_y']
    rz = pitch_data['release_pos_z']
    velo = pitch_data['release_speed']
    y0 = 50
    yf = 17/12

    # vertical and horizontal release angle
    theta_z0 = -np.arctan(vz0/vy0)*180/np.pi
    theta_x0 = -np.arctan(vx0/vy0)*180/np.pi
    pitch_data['release_angle_v'] = round(theta_z0, 2)
    pitch_data['release_angle_h'] = round(theta_x0, 2)

    # vertical and horizontal approach angle
    vyf = -np.sqrt(vy0**2- (2 * ay * (y0 - yf)))
    t = (vyf - vy0)/ay
    vzf = vz0 + (az*t)
    vxf = vx0 + (ax*t)

    theta_zf = -np.arctan(vzf/vyf)*180/np.pi
    theta_xf = -np.arctan(vxf/vyf)*180/np.pi
    pitch_data['VAA'] = round(theta_zf, 2)
    pitch_data['HAA'] = round(theta_xf, 2)

    # my calculations of VAA and HAA
    zf = pitch_data['plate_z']
    delta_z = rz - zf
    delta_y = ry - yf
    phi_z = -np.arctan(delta_z/delta_y)*180/np.pi

    xf = pitch_data['plate_x']
    delta_x = rx - xf
    phi_x = -np.arctan(delta_x/delta_y)*180/np.pi

    pitch_data.insert(89, 'my_VAA', round(2*phi_z - theta_z0, 2))
    pitch_data.insert(91, 'my_HAA', round(2*phi_x - theta_x0, 2))

    delta = []
    for v in range(75,102):
        condition = 'release_speed > @v - 0.5 and release_speed < @v + 0.5'
        VAA = pitch_data.query(condition)['VAA'].mean()
        my_VAA = pitch_data.query(condition)['my_VAA'].mean()
        delta.append(VAA - my_VAA)
        
    model = np.poly1d(np.polyfit(np.arange(75,102), delta, 2))

    velo_correction = model[0] + model[1]*velo + model[2]*velo**2
    geom_VAA = 2*phi_z - theta_z0 + velo_correction
    geom_HAA = 2*phi_x - theta_x0 + velo_correction

    pitch_data.drop(columns=['my_VAA', 'my_HAA'], inplace=True)
    pitch_data.insert(89, 'geom_VAA', round(geom_VAA, 2))
    pitch_data.insert(91, 'geom_HAA', round(geom_HAA, 2))

    # total break angle
    delta_theta_z = theta_z0 - theta_zf
    delta_theta_x = theta_x0 - theta_xf
    delta_theta = np.sqrt(delta_theta_z**2 + delta_theta_x**2)
    pitch_data['break_angle'] = round(delta_theta, 2)

    # sharpness of break
    eff_t = delta_y/velo
    iVB = pitch_data['pfx_z']
    sharpness = np.abs(iVB/eff_t)
    pitch_data['sharpness'] = round(sharpness, 2)
    
    #
    # assign binary values for outcomes of interest
    #
    
    swstr_types = ['swinging_strike_blocked', 'swinging_strike', 'foul_tip']
    swing_tpyes = swstr_types + ['foul', 'hit_into_play']
    strike_zones = np.arange(1,10)
    pitch_data['zone_value'] = pitch_data['zone'].isin(strike_zones).astype(int)
    pitch_data['swing_value'] = pitch_data['description'].isin(swing_tpyes).astype(int)
    pitch_data['swstr_value'] = pitch_data['description'].isin(swstr_types).astype(int)
    pitch_data['bip_value'] = (pitch_data['description'] == 'hit_into_play').astype(int)
    pitch_data['gb_value'] = (pitch_data['bb_type'] == 'ground_ball').astype(int)
    pitch_data['fb_value'] = ((pitch_data['description'] == 'hit_into_play') & 
                                      (pitch_data['bb_type'] == 'fly_ball')).astype(int)
    
    return pitch_data
pitch_data = get_data()

# function for pitch level pairwise analysis
def pitch_level_analysis(data, master_df, who, statistic, colname=None, diff_type='raw', factor=None, pitch_types='all', conditions=None, min_events=10, variance_threshold=0.1):
        
    print()
    print('Preparing data...')
    
    #
    # Ensure proper entries
    #
    
    if type(data) != pd.core.frame.DataFrame:
        raise TypeError('Please emsure "data" is a Pandas dataframe.')
    if type(master_df) != pd.core.frame.DataFrame:
        raise TypeError('Please emsure "master_df" is a Pandas dataframe.')
    if who not in ['pitcher', 'batter']:
        raise ValueError('The variable "who" takes one of two values: pitcher or batter.')
    if statistic not in data.columns:
        raise ValueError('The variable "statistic" must be a named column in "data."')
    if colname is not None and type(colname) != str:
        raise ValueError('The variable "colname" must have type None or str.')
    if diff_type not in ['raw', 'percent', 'z-score']:
        raise ValueError('The variable "diff_type" takes one of three values: raw, percent, or z-score')
    if factor is not None and type(factor) not in [int, float]:
        raise ValueError('The variable "factor" must have type None or int or float.')
    if pitch_types not in ['all', 'each'] and not all(p in data['pitch_type'].unique() for p in pitch_types):
        raise ValueError('Invalid pitch type abbreviation.')
    if conditions is not None and type(conditions) != str:
        raise ValueError('The variable "conditions" must have type None or str.')
    
    #
    # Apply inputs
    #
    
    data = data.dropna(subset = [statistic]).copy()
    
    # scale factor
    if factor is not None:
        data[statistic] = (data[statistic]*factor).copy()
    
    # apply conditions
    if conditions is not None:
        data = data.query(conditions).copy()
    
    # all pitches of this type
    if pitch_types in  ['all', 'each']:
        pitch_counts = data.groupby('pitch_type').size()
        valid_pitch_types = (pitch_counts[pitch_counts > 1000]).index
        data = data[data['pitch_type'].isin(valid_pitch_types)].copy()
    else:
        data = data[data['pitch_type'].isin(pitch_types)].copy()
    
    # baselines
    if pitch_types != 'each':
        stat_mean = np.abs(data[statistic].mean())
        stat_std = data.groupby(who)[statistic].mean().std()
        stat_iqr = np.percentile(data.groupby(who)[statistic].mean(), 75) - np.percentile(data.groupby(who)[statistic].mean(), 25)

    else:
        stat_mean = {}
        stat_std = {}
        stat_iqr = data.groupby('pitch_type')[statistic].apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        for pitch_type in valid_pitch_types:
            
            pitch_type_data = data.loc[data['pitch_type'] == pitch_type, [who, statistic]]
            
            pitcher_counts = pitch_type_data.groupby(who).size()
            valid_pitchers = pitcher_counts[pitcher_counts > 100].index
            valid_data = pitch_type_data[pitch_type_data[who].isin(valid_pitchers)]
            
            stat_mean[pitch_type] = np.abs(valid_data[statistic].mean())
            stat_std[pitch_type] = valid_data.groupby('pitcher')[statistic].std().mean()
            
        data['stat_mean'] = data['pitch_type'].map(stat_mean)
    
    #
    # Prep for analysis
    #
    
    # generate all possible pairs of ballparks
    ballparks = np.sort(pitch_data['home_team'].unique())
    ballpark_pairs = list(combinations(ballparks, 2))
    
    # loop over pairs
    print()
    players_used = []
    if pitch_types != 'each':
        
        # dataframe to store park effects for this statistic
        mean_deltas = pd.DataFrame(columns=list(ballparks), index=ballparks, dtype=float)
    
        for park1, park2 in tqdm(ballpark_pairs):
            
            # filter to only include events at the two parks
            filtered_by_park = data[data['home_team'].isin([park1, park2])]
            
            # group by player and park and count the number of events
            event_counts = filtered_by_park.groupby([who, 'home_team']).size().unstack(fill_value=0)
           
            # filter player pitch types who have experienced enough events at both parks
            enough_events = event_counts[(event_counts[park1] >= min_events) & (event_counts[park2] >= min_events)].index
            players_with_enough_events = filtered_by_park[filtered_by_park.set_index(who).index.isin(enough_events)]        
            
            # filter players with low standard error compared to park-pitch type variance
            player_event_error = players_with_enough_events.groupby([who, 'home_team']).sem(numeric_only=True)[statistic]
            normalized_error = player_event_error / stat_iqr
            consistent_players = normalized_error[normalized_error < variance_threshold]
            passes_both_parks = consistent_players.groupby(who).filter(lambda x: len(x) > 1).reset_index(level='home_team', drop=True)
            if len(passes_both_parks) < 5:
                raise ValueError('Insufficient number of players satisfying variance conditions. Try increasing variance_threshold.')

            # get data for good players
            players_used.append(len(passes_both_parks))
            player_data = filtered_by_park[filtered_by_park[who].isin(passes_both_parks.index)].copy()
            
            # aggregate by player and park
            stat_player_park = player_data.groupby([who, 'home_team'])[statistic].mean().reset_index()
    
            # unweighted average of statistic for each player
            unweighted_avg_stat = stat_player_park.groupby(who)[statistic].mean().reset_index()
            unweighted_avg_stat.rename(columns={statistic: 'unweighted_avg_stat'}, inplace=True)
            stat_player_park = pd.merge(stat_player_park, unweighted_avg_stat, on=who, how='left')
    
            # find difference between parks
            if diff_type == 'raw':
                stat_player_park['delta'] = stat_player_park[statistic] - stat_player_park['unweighted_avg_stat']
            elif diff_type == 'percent':
                stat_player_park['delta'] = stat_player_park[statistic]/stat_player_park['unweighted_avg_stat']
            else:
                # inter-park stdev for each pitch type
                grouped_std = player_data.groupby([who, 'home_team'])[statistic].std()
                inter_park_std = grouped_std.groupby('home_team').mean()
                stat_player_park['inter_park_std'] = stat_player_park.set_index(['home_team']).index.map(inter_park_std)
                
                # z-scores for each pitch based on player baseline and park-pitch type stdev
                stat_player_park['delta'] = (stat_player_park[statistic] - stat_player_park['unweighted_avg_stat']) / stat_player_park['inter_park_std']    
    
            # calculate mean delta for each park
            mean_delta_by_park = stat_player_park.groupby('home_team')['delta'].mean()
    
            # store the results
            mean_deltas.loc[park1, park2] = mean_delta_by_park[0]
            mean_deltas.loc[park2, park1] = mean_delta_by_park[1]
    
        # aggregate means and errors for each park
        park_mean = mean_deltas.mean(axis=1)
        park_stderr = mean_deltas.sem(axis=1)
    
    else:
        
        # dataframe to store park effects for this statistic
        index = pd.MultiIndex.from_product([ballparks, ballparks, valid_pitch_types], names=['park1', 'park2', 'pitch_type'])
        mean_deltas = pd.DataFrame(np.nan, index=index, columns=['value'])
        
        for park1, park2 in tqdm(ballpark_pairs):
        
            # filter to only include events at the two parks
            filtered_by_park = data[data['home_team'].isin([park1, park2])]
            
            # group by player, pitch type, and park and count the number of events
            event_counts = filtered_by_park.groupby([who, 'pitch_type', 'home_team']).size().unstack(fill_value=0)
    
            # filter player pitch types who have experienced enough events at both parks
            enough_events = event_counts[(event_counts[park1] >= min_events) & (event_counts[park2] >= min_events)].index
            players_with_enough_events = filtered_by_park[filtered_by_park.set_index([who, 'pitch_type']).index.isin(enough_events)]        
            
            # filter players with low standard error compared to park-pitch type variance
            player_event_error = players_with_enough_events.groupby([who, 'pitch_type', 'home_team']).sem(numeric_only=True)[statistic]
            normalized_error = player_event_error / player_event_error.index.get_level_values('pitch_type').map(stat_iqr)
            consistent_player_pitch_types = normalized_error[normalized_error < variance_threshold]
            passes_both_parks = consistent_player_pitch_types.groupby([who, 'pitch_type']).filter(lambda x: len(x) > 1).reset_index(level='home_team', drop=True)
            if len(passes_both_parks) < 5:
                raise ValueError('Insufficient number of player pitch types satisfying variance conditions. Try increasing variance_threshold.')

            # get data for good player & pitch types
            players_used.append(len(passes_both_parks))
            player_data = filtered_by_park[filtered_by_park.set_index([who, 'pitch_type']).index.isin(passes_both_parks.index)].copy()
    
            # aggregate by player, pitch type, and park
            stat_player_park = player_data.groupby([who, 'pitch_type', 'home_team'])[statistic].mean().reset_index()
    
            # unweighted average of statistic for each player & pitch type
            unweighted_avg_stat = stat_player_park.groupby([who, 'pitch_type'])[statistic].mean().reset_index()
            unweighted_avg_stat.rename(columns={statistic: 'unweighted_avg_stat'}, inplace=True)
            stat_player_park = pd.merge(stat_player_park, unweighted_avg_stat, on=[who, 'pitch_type'], how='left')

            # find difference between parks
            if diff_type == 'raw':
                stat_player_park['delta'] = stat_player_park[statistic] - stat_player_park['unweighted_avg_stat']
            elif diff_type == 'percent':
                stat_player_park['delta'] = stat_player_park[statistic]/stat_player_park['unweighted_avg_stat']
            else:
                # inter-park stdev for each pitch type
                grouped_std = player_data.groupby([who, 'pitch_type', 'home_team'])[statistic].std()
                inter_park_std = grouped_std.groupby(['pitch_type', 'home_team']).mean()
                stat_player_park['inter_park_std'] = stat_player_park.set_index(['pitch_type', 'home_team']).index.map(inter_park_std)
                
                # z-scores for each pitch based on player baseline and park-pitch type stdev
                stat_player_park['delta'] = (stat_player_park[statistic] - stat_player_park['unweighted_avg_stat']) / stat_player_park['inter_park_std']    
    
            # calculate mean delta for each park
            mean_delta_by_park = stat_player_park.groupby(['home_team', 'pitch_type'])['delta'].mean()
    
            # store the results
            mean_deltas.loc[(park1, park2), :] = mean_delta_by_park.loc[park1].reindex(valid_pitch_types).values
            mean_deltas.loc[(park2, park1), :] = mean_delta_by_park.loc[park2].reindex(valid_pitch_types).values
        
        # aggregate means and errors for each park
        park_mean = mean_deltas.groupby(level=0).mean()
        park_stderr = mean_deltas.groupby(level=0).sem()
        
        # formatting
        park_mean.index.name = None
        park_stderr.index.name = None
        park_mean = park_mean['value']
        park_stderr = park_stderr['value']

    # min and max diff
    max_d = park_mean.max()
    min_d = park_mean.min()
    
    # min and max z-score
    if pitch_types != 'each':
        
        if diff_type == 'raw':
            max_z = NormalDist(mu=0, sigma=stat_std).zscore(max_d)
            min_z = NormalDist(mu=0, sigma=stat_std).zscore(min_d)
            effect = round((max_z - min_z)/2, 2)
            
        elif diff_type == 'percent':
            max_z = NormalDist(mu=1, sigma=stat_std).zscore(max_d)
            min_z = NormalDist(mu=1, sigma=stat_std).zscore(min_d)
            effect = round((max_z - min_z)/2, 2)
        
        else:
            max_z = max_d
            min_z = min_d
            
    else:
        
        if diff_type == 'raw':
            max_z = np.nanmean(list({p: NormalDist(mu=0, sigma=stat_std[p]).zscore(max_d) for p in stat_std}.values()))
            min_z = np.nanmean(list({p: NormalDist(mu=0, sigma=stat_std[p]).zscore(min_d) for p in stat_std}.values()))
        
        elif diff_type == 'percent':
            percent_std = {p: stat_std[p]/stat_mean[p] for p in stat_mean}
            max_z = np.nanmean(list({p: NormalDist(mu=0, sigma=percent_std[p]).zscore(max_d) for p in percent_std}.values()))
            min_z = np.nanmean(list({p: NormalDist(mu=0, sigma=percent_std[p]).zscore(min_d) for p in percent_std}.values()))
        
        else:
            max_z = max_d
            min_z = min_d
        
    effect = np.round((max_z - min_z)/2, 2)
    
    # add to master df
    if colname is None:
        colname = statistic
    if effect > 0.01:
        master_df[colname] = [f"{effect:.2f}"] + list(park_mean)
    
    # tell the user the results
    if diff_type == 'raw':
        diff = round((max_d - min_d)/2, 2)
        
        if 'speed' in colname:
            units = ' MPH'
        elif 'AA' in colname:
            units = ' degrees'
        elif '%' in colname:
            units = ' percentage points'
        else:
            units = ' inches'
        
        if factor == 12:
            units = ' inches'
            
    elif diff_type == 'percent':
        diff = round(100*(max_d - min_d)/2, 2)
        units = '%'
    
    else:
        diff = effect
        units = ' standard deviations'
       
    # display
    print()
    if pitch_types != 'each':
        print('Minimum', np.min(players_used), who + 's per park pair.')
    
    else:
        print('Minimum', np.min(players_used), 'player pitch types per park pair.')
    
    if diff_type != 'z-score':
        if sum(letter.isupper() for letter in colname) < 2:
            print(colname.title() + ' varies by +/-' + str(diff) + units + ' (' + str(effect) + ' standard deviations) due to park effects.')   
        else:
            print(colname + ' varies by +/-' + str(diff) + units + ' (' + str(effect) + ' standard deviations) due to park effects.')
    else:
        if sum(letter.isupper() for letter in colname) < 2:
            print(colname.title() + ' varies by +/-' + str(diff) + units + ' due to park effects.')   
        else:
            print(colname + ' varies by +/-' + str(diff) + units + ' due to park effects.')
    print()    
    
    # return dataframe with results for plotting
    park_grades = pd.concat([park_mean, park_stderr], axis = 1)
    park_grades.columns = ['park_mean', 'park_stderr']
    
    return park_grades

# function for PA level pairwise analysis
def PA_level_analysis(data, master_df, statistic, colname=None, diff_type='raw', factor=None, pitch_types='all', conditions=None, min_events=None, variance_threshold=None):
        
    print()
    print('Preparing data...')
    
    #
    # Ensure proper entries
    #
    
    if type(data) != pd.core.frame.DataFrame:
        raise TypeError('Please emsure "data" is a Pandas dataframe.')
    if type(master_df) != pd.core.frame.DataFrame:
        raise TypeError('Please emsure "master_df" is a Pandas dataframe.')
    if statistic not in data.events.values:
        raise ValueError('The variable "statistic" must be a named event in data.events.')
    if colname is not None and type(colname) != str:
        raise ValueError('The variable "colname" must have type None or str.')
    if diff_type not in ['raw', 'percent', 'z-score']:
        raise ValueError('The variable "diff_type" takes one of three values: raw, percent, or z-score')
    if factor is not None and type(factor) not in [int, float]:
        raise ValueError('The variable "factor" must have type None or int or float.')
    if pitch_types != 'all' and not all(p in data['pitch_type'].unique() for p in pitch_types):
        raise ValueError('Invalid pitch type abbreviation.')
    if conditions is not None and type(conditions) != str:
        raise ValueError('The variable "conditions" must have type None or str.')
    
    #
    # Apply inputs
    #
    
    # PA level
    data = data.dropna(subset=['events']).copy()
    data[statistic] = (data['events'] == statistic).astype(int)
    if min_events is None:
        min_events = 10
    if variance_threshold is None:
        variance_threshold = 1
    
    # apply conditions
    if conditions is not None:
        data = data.query(conditions).copy()
    
    # all pitches of this type
    if pitch_types != 'all':
        data = data[data['pitch_type'].isin(pitch_types)].copy()
    else:
        pitch_counts = data.groupby('pitch_type').size()
        valid_pitch_types = (pitch_counts[pitch_counts > 1000]).index
        data = data[data['pitch_type'].isin(valid_pitch_types)].copy()
        
    # scale factor
    if factor is not None:
        data[statistic] = (data[statistic]*factor).copy()
        
    #
    # Prep for analysis
    #
    
    pitcher_std = data.groupby('pitcher')[statistic].mean().std()
    pitcher_iqr = np.percentile(data.groupby('pitcher')[statistic].mean(), 75) - np.percentile(data.groupby('pitcher')[statistic].mean(), 25)
    
    batter_std = data.groupby('batter')[statistic].mean().std()
    batter_iqr = np.percentile(data.groupby('batter')[statistic].mean(), 75) - np.percentile(data.groupby('batter')[statistic].mean(), 25)
    
    # create interactive baseline
    test_data = data.groupby('pitcher').filter(lambda x: len(x) >= 100)
    test_data = data.groupby('batter').filter(lambda x: len(x) >= 100)

    pitcher_stat = test_data.loc[:, ['pitcher', statistic]].groupby('pitcher').mean(numeric_only=True)[statistic]
    batter_stat = test_data.loc[:, ['batter', statistic]].groupby('batter').mean(numeric_only=True)[statistic]

    test_data['batter_stat'] = test_data['batter'].map(batter_stat)
    test_data['pitcher_stat'] = test_data['pitcher'].map(pitcher_stat)

    # logistic regression
    x = test_data[['batter_stat', 'pitcher_stat']]
    y = test_data[statistic]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    def pred_stat(matchup_data):
        input_data = matchup_data[['batter_stat', 'pitcher_stat']]
        return log_reg.predict_proba(input_data)[:, 1]
    
    # generate all possible pairs of ballparks
    ballparks = np.sort(pitch_data['home_team'].unique())
    ballpark_pairs = list(combinations(ballparks, 2))
    
    # loop over pairs
    print()
    players_used = []
        
    # dataframe to store park effects for this statistic
    mean_deltas = pd.DataFrame(columns=list(ballparks), index=ballparks, dtype=float)

    for park1, park2 in tqdm(ballpark_pairs):
            
        # filter to only include events at the two parks
        filtered_by_park = data[data['home_team'].isin([park1, park2])]
        
        #
        # find pitcher baselines
        #
        
        # group by pitcher and park and count the number of events
        pitcher_counts = filtered_by_park.groupby(['pitcher', 'home_team']).size().unstack(fill_value=0)
       
        # filter pitchers who have experienced enough events at both parks
        enough_pitcher = pitcher_counts[(pitcher_counts[park1] >= min_events) & (pitcher_counts[park2] >= min_events)].index
        pitchers_with_enough_events = filtered_by_park[filtered_by_park.set_index('pitcher').index.isin(enough_pitcher)]        
        
        # filter pitchers with low standard error compared to league variance
        pitcher_event_error = pitchers_with_enough_events.groupby(['pitcher', 'home_team']).sem(numeric_only=True)[statistic]
        normalized_p_error = pitcher_event_error / pitcher_iqr
        consistent_pitchers = normalized_p_error[(normalized_p_error < variance_threshold) & (normalized_p_error > 0)]
        pitchers_both_parks = consistent_pitchers.groupby('pitcher').filter(lambda x: len(x) > 1).reset_index(level='home_team', drop=True)

        # get data for good pitchers
        pitcher_data = filtered_by_park[filtered_by_park['pitcher'].isin(pitchers_both_parks.index)].copy()
        
        # aggregate by pitcher and park
        stat_pitcher_park = pitcher_data.groupby(['pitcher', 'home_team'])[statistic].mean().reset_index()
        stat_pitcher_park.rename(columns = {statistic:'pitcher_stat'}, inplace=True)
        

        #
        # find batter baselines
        #

        # group by batter and park and count the number of events
        batter_counts = filtered_by_park.groupby(['batter', 'home_team']).size().unstack(fill_value=0)
       
        # filter batters who have experienced enough events at both parks
        enough_batter = batter_counts[(batter_counts[park1] >= min_events) & (batter_counts[park2] >= min_events)].index
        batters_with_enough_events = filtered_by_park[filtered_by_park.set_index('batter').index.isin(enough_batter)]        
        
        # filter batters with low standard error compared to league variance
        batter_event_error = batters_with_enough_events.groupby(['batter', 'home_team']).sem(numeric_only=True)[statistic]
        normalized_b_error = batter_event_error / batter_iqr
        consistent_batters = normalized_b_error[(normalized_b_error < variance_threshold) & (normalized_b_error > 0)]
        batters_both_parks = consistent_batters.groupby('batter').filter(lambda x: len(x) > 1).reset_index(level='home_team', drop=True)

        # get data for good batters
        batter_data = filtered_by_park[filtered_by_park['batter'].isin(batters_both_parks.index)].copy()
        
        # aggregate by batter and park
        stat_batter_park = batter_data.groupby(['batter', 'home_team'])[statistic].mean().reset_index()
        stat_batter_park.rename(columns = {statistic:'batter_stat'}, inplace=True)
        
        #
        # find interactive baselines
        #

        matchup_data = filtered_by_park[(filtered_by_park['batter'].isin(batters_both_parks.index)) & 
                                        (filtered_by_park['pitcher'].isin(pitchers_both_parks.index))].copy()
        matchup_data = matchup_data.merge(stat_batter_park[['batter', 'home_team', 'batter_stat']], 
                                          on=['batter', 'home_team'], 
                                          how='left')
        matchup_data = matchup_data.merge(stat_pitcher_park[['pitcher', 'home_team', 'pitcher_stat']], 
                                          on=['pitcher', 'home_team'], 
                                          how='left')
        matchup_data['expected_stat'] = pred_stat(matchup_data)
        players_used.append(len(matchup_data))

        # find difference between parks
        p1_data = matchup_data.query('home_team == @park1')
        p2_data = matchup_data.query('home_team == @park2')

        if diff_type == 'raw':
            delta1 = p1_data['expected_stat'].mean() - matchup_data['expected_stat'].mean()
            delta2 = p2_data['expected_stat'].mean() - matchup_data['expected_stat'].mean()
        elif diff_type == 'percent':
            delta1 = p1_data['expected_stat'].mean() / matchup_data['expected_stat'].mean()
            delta2 = p2_data['expected_stat'].mean() / matchup_data['expected_stat'].mean()

        # store the results
        mean_deltas.loc[park1, park2] = delta1
        mean_deltas.loc[park2, park1] = delta2

    # aggregate means and errors for each park
    park_mean = mean_deltas.mean(axis=1)
    park_stderr = mean_deltas.sem(axis=1)

    # min and max diff
    max_d = park_mean.max()
    min_d = park_mean.min()
    
    # min and max z-score
    stat_std = (pitcher_std + batter_std)/2
    if diff_type == 'raw':
        max_z = NormalDist(mu=0, sigma=stat_std).zscore(max_d)
        min_z = NormalDist(mu=0, sigma=stat_std).zscore(min_d)
        effect = round((max_z - min_z)/2, 2)
        
    elif diff_type == 'percent':
        max_z = NormalDist(mu=1, sigma=stat_std).zscore(max_d)
        min_z = NormalDist(mu=1, sigma=stat_std).zscore(min_d)
        effect = round((max_z - min_z)/2, 2)
    
    else:
        max_z = max_d
        min_z = min_d
        
    z = (max_z - min_z)/2
    effect = round(z, -int(floor(log10(z))) + 1)
    
    # add to master df
    if colname is None:
        colname = statistic
    if effect > 0.01:
        master_df[colname] = [f"{effect:.2f}"] + list(park_mean)
    
    # tell the user the results
    if diff_type == 'raw':
        d = (max_d - min_d)/2
        diff = round(d, -int(floor(log10(z))) + 1)
        
        if 'speed' in colname:
            units = ' MPH'
        elif 'AA' in colname:
            units = ' degrees'
        elif '%' in colname:
            units = ' percentage points'
        else:
            units = ' inches'
        
        if factor == 12:
            units = ' inches'
            
    elif diff_type == 'percent':
        d = 100*(max_d - min_d)/2
        diff = round(d, -int(floor(log10(z))) + 1)
        units = '%'
    
    else:
        diff = effect
        units = ' standard deviations'
       
    # display
    print()
    print('Minimum', np.min(players_used), 'matchups per park pair.')
    
    if diff_type != 'z-score':
        if sum(letter.isupper() for letter in colname) < 2:
            print(colname.title() + ' varies by +/-' + str(diff) + units + ' (' + str(effect) + ' standard deviations) due to park effects.')   
        else:
            print(colname + ' varies by +/-' + str(diff) + units + ' (' + str(effect) + ' standard deviations) due to park effects.')
    else:
        if sum(letter.isupper() for letter in colname) < 2:
            print(colname.title() + ' varies by +/-' + str(diff) + units + ' due to park effects.')   
        else:
            print(colname + ' varies by +/-' + str(diff) + units + ' due to park effects.')
    print()    
    
    # return dataframe with results for plotting
    park_grades = pd.concat([park_mean, park_stderr], axis = 1)
    park_grades.columns = ['park_mean', 'park_stderr']
    
    return park_grades

# function for making park-by-park plot with error bars
def park_plot(y, title, ylab, sort = 'descending', figsize=(10,5), s=100, color_scale=1):
    
    # make figure
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    # color points based on the values
    norm = plt.Normalize(y['park_mean'].min(), y['park_mean'].max())
    
    # sort
    if sort == 'alphabetical':
        y = y.copy().sort_index()
    if sort == 'descending':
        y = y.copy().sort_values(by='park_mean', ascending=False)
    if sort == 'ascending':
        y =y.copy().sort_values(by='park_mean', ascending=True)
        
    # scatterplot
    plt.scatter(
        y.index,
        y['park_mean'],
        c=y['park_mean']*color_scale,
        cmap='coolwarm',
        norm=norm,
        s=s)
    
    # make error bars
    plt.errorbar(y.index, y['park_mean'], 
                 yerr=y['park_stderr'],
                 fmt='none',
                 ecolor='grey',
                 elinewidth=2,
                 capsize=4)
    
    # formatting
    ax.set_title(title)
    ax.set_xlabel('Home Team')
    ax.set_ylabel(ylab)
    ax.tick_params(axis='x', rotation=45)
    
    # text
    y_limits = ax.get_ylim()
    bot = y_limits[0] + 0.1*(y_limits[1] - y_limits[0])
    top = y_limits[1] - 0.1*(y_limits[1] - y_limits[0])
    ax.text(20, top, 'Data via Baseball Savant, 2021-2024')
    ax.text(15, bot, 'Error bars represent standard error (N = 30)', ha='center', va='top')
    
    # percent if necessary
    if '%' in ylab:
        if (y['park_mean'].max() - y['park_mean'].min()) > 0.1:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.0f}%'))
        else:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.1f}%'))

    
    plt.tight_layout()
    plt.show()

# function for making scatterplot
def scatterplot(x, y, title, xlab, ylab, drop_rows=None, xticks=None, xtlabs=None, xrot=None, yticks=None, ytlabs=None, figsize=None, loc='bot', s=100, trendline=False, x_extrema_labels=False, y_extrema_labels=False, plot_type='qt'):
    
    x = x.sort_index().copy()
    y = y.sort_index().copy()
    
    if 'effect' in x.index:
        x = x.drop('effect').astype('float64')
    if 'effect' in y.index:
        y = y.drop('effect').astype('float64')

    if drop_rows:
        x, y = x.drop(drop_rows), y.drop(drop_rows)
    
    if plot_type == 'qt':
        ipython = get_ipython()
        ipython.run_line_magic('matplotlib', 'qt')
    if plot_type == 'inline':
        ipython = get_ipython()
        ipython.run_line_magic('matplotlib', 'inline')
    
    # make plot
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
        
    ax.scatter(x, y, s = s)

    # formatting
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    ax.set_ylim((y_limits[0] - 0.1*(y_limits[1] - y_limits[0])), (y_limits[1] + 0.1*(y_limits[1] - y_limits[0])))
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if xticks is not None:
        ax.set_xticks(xticks)
        if xtlabs is not None:
            ax.set_xticklabels(xtlabs)
            
    if xrot is not None:
        ax.tick_params(axis='x', rotation=xrot)
        
    if yticks is not None:
        ax.set_yticks(yticks)
        if ytlabs is not None:
            ax.set_yticklabels(ytlabs)

    y_limits = ax.get_ylim()
    mid_x = np.mean(x_limits)
    
    if loc == 'bot':
        y_pos = y_limits[0] + (y_limits[1] - y_limits[0]) * 0.05
    if loc == 'top':
        y_pos = y_limits[0] + (y_limits[1] - y_limits[0]) * 0.95

    ax.text(mid_x, y_pos, 'Data via Baseball Savant, 2021-2024', ha='center', va='top', fontsize=10)
    
    # trendline
    if trendline:
        m, b, r, p, std_err = stats.linregress(x, y)
        plt.plot([x.min(), x.max()],
                 [x.min()*m + b, x.max()*m + b],
                 '--', color = 'gray')
        
        y_pos = y_limits[0] + (y_limits[1] - y_limits[0]) * 0.9
        
        if m > 0:
            x_pos = x_limits[0] + (x_limits[1] - x_limits[0]) * 0.1
            ax.text(x_pos, y_pos, f'$R^2$ = {round(r**2, 2)}', ha='left', va='top', fontsize = 12)
        else:
            x_pos = x_limits[0] + (x_limits[1] - x_limits[0]) * 0.9
            ax.text(x_pos, y_pos, f'$R^2$ = {round(r**2, 2)}', ha='right', va='top', fontsize = 12)
    
    if x_extrema_labels:
        
        highest_x_idx = x.idxmax()
        lowest_x_idx = x.idxmin()
        offset = 0.05*(y_limits[1] - y_limits[0])

        # highest mound
        plt.text(x[highest_x_idx], y[highest_x_idx] + offset, highest_x_idx, 
                  fontsize=12, ha='center', color='black', weight='bold')
        # lowest mound
        plt.text(x[lowest_x_idx], y[lowest_x_idx] + offset, lowest_x_idx,
                  fontsize=12, ha='center', color='black', weight='bold')


    if y_extrema_labels:
        
        highest_y_idx = y.idxmax()
        lowest_y_idx = y.idxmin()
        offset = 0.05*(y_limits[1] - y_limits[0])

        # highest mound
        plt.text(x[highest_y_idx], y[highest_y_idx] + offset, highest_y_idx, 
                  fontsize=12, ha='center', color='black', weight='bold')
        # lowest mound
        plt.text(x[lowest_y_idx], y[lowest_y_idx] + offset, lowest_y_idx,
                  fontsize=12, ha='center', color='black', weight='bold')
    
    # percent if necessary
    if '%' in ylab:
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100*y:.1f}%'))
    
    plt.tight_layout()
    plt.show()



'''
###############################################################################
################################# Park Effects ################################
###############################################################################
'''



# make master park effects dataframe
park_effects = pd.DataFrame(index=['effect'] + list(np.sort(pitch_data['home_team'].unique())))



#
# Example Pitch-Level Effects
#

# release height
rh_by_park = pitch_level_analysis(data = pitch_data,
                                  master_df = park_effects,
                                  who = 'pitcher',
                                  statistic = 'release_pos_z',
                                  colname = 'release_height',
                                  factor = 12)
park_plot(y = rh_by_park,
          title = 'Ballpark Effects on Release Height',
          ylab = 'Release Height Added (inches)')



# release extension
ext_by_park = pitch_level_analysis(data = pitch_data,
                                   master_df = park_effects,
                                   who = 'pitcher',
                                   statistic = 'release_extension',
                                   factor = 12)
park_plot(y = ext_by_park,
          title = 'Ballpark Effects on Extension',
          ylab = 'Extension Added (inches)')



#
# Example Pitch Type-Level Effects
#

# glove side break
glove_HB_by_park = pitch_level_analysis(data = pitch_data,
                                        master_df = park_effects,
                                        who = 'pitcher',
                                        statistic = 'pfx_x',
                                        colname = 'HB_glove',
                                        pitch_types = 'each',
                                        diff_type = 'percent',
                                        conditions = 'pfx_x < -3',
                                        factor = -12)
park_plot(y = glove_HB_by_park,
          title = 'Ballpark Effects on Glove Side Break',
          ylab = 'Horizontal Break Effect (%)')



# VAA
VAA_by_park = pitch_level_analysis(data = pitch_data,
                                   master_df = park_effects,
                                   who = 'pitcher',
                                   statistic = 'VAA',
                                   diff_type = 'percent',
                                   pitch_types = 'each')
park_plot(y=VAA_by_park,
          title='Ballpark Effects on Vertical Approach Angle',
          ylab='VAA Effect (standard deviations)')



#
# Example PA-Level Effects
#

# K%
k_by_park = PA_level_analysis(data = pitch_data,
                                  master_df = park_effects,
                                  statistic = 'strikeout',
                                  colname = 'k%',
                                  diff_type = 'percent')
park_plot(y=k_by_park,
          title='Relative Ballpark Effects on K%',
          ylab='K% Effect')



# BB%
bb_by_park = PA_level_analysis(data = pitch_data,
                                   master_df = park_effects,
                                   statistic = 'walk',
                                   colname = 'bb%',
                                   diff_type = 'percent')
park_plot(y=bb_by_park,
          title='Relative Ballpark Effects on BB%',
          ylab='BB% Effect')



#
# Rank Park Effects by Magnitude (work in progress - not accurate between types)
#

print(park_effects.loc['effect'].sort_values(ascending=False))



'''
###############################################################################
########################## Park Effect Relationships ##########################
###############################################################################
'''



# release height vs extension
scatterplot(x=park_effects['release_height'], 
            y=park_effects['release_extension'],
            title='Release Height Added vs. Extension Added by Park',
            xlab='Release Height Added (inches)', 
            ylab='Extension Added by Park (inches)',
            s=50,
            trendline=False,
            x_extrema_labels=True,
            y_extrema_labels=True)


# release height vs VAA
scatterplot(x=park_effects['release_height'], 
            y=park_effects['VAA'],
            title='Release Height Added vs. VAA Added by Park',
            xlab='Release Height Added (inches)', 
            ylab='VAA Added (degrees)',
            s=50,
            trendline=True,
            x_extrema_labels=True,
            y_extrema_labels=True)


# K% vs BB%
scatterplot(x=park_effects['k%'], 
            y=park_effects['bb%'],
            title='Strikoeuts Added vs. Walks Added by Park',
            xlab='Strikeout Effect (%)', 
            ylab='Walk Effect (%)',
            s=50,
            trendline=False,
            x_extrema_labels=True,
            y_extrema_labels=True)
