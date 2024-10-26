from test_regression import *
from code_to_category import *
import pandas as pd

#This code uses two-way raking to determine geographic units' survey weights from a national survey
#First a matrix of weights is obtained by raking the national survey for each geographic unit,
#where columns correspond to the survey weights in each geographic unit
#The quotient between the national survey weights and the summed weights gives the vector that adjusts
#the geographic unit weights so that they sum to the national marginals.
#We multiply the geographic weight matrix by this vector to find the final weights for each geographic unit.

regress_data = pd.read_csv('model_data/regression_data_altered.csv')
regress_data = regress_data.set_index(['year','state'])
#code for determining average ideology of state electorates over time
cand_positions = pd.read_csv('model_data/candidate_positions_unweighted.csv')
cand_positions = cand_positions.set_index('year')

for yr in range(1972,2000,4):
    df_yr = pd.read_csv('model_data/simulation_data.csv')
    df_yr = df_yr[df_yr['year'] == yr]
    rake_keys = list(census_keys.keys())[:-1]
    state_weights = pd.DataFrame()

    #Find weights of each respondent within state by raking
    state_pop = states[['year','Persons: Total']].dropna()
    state_df = dict()
    total_votes = dict()

    for state in state_postal_codes:
        fips = code_to_fips[state]
        state_df[state] = rake_state(yr, fips, df_yr, rake_keys)
        print(f'raked ({yr},{state})')
        df = state_df[state]
        
        #code for determining average ideology of state electorates over time
        state_ideo = sum(df['ideology'] * df['weight1']) / sum(df['weight1'])
        regress_data.loc[(yr,state), 'ideology'] = state_ideo

        diff_dem = abs(state_ideo - cand_positions.loc[yr,'dem_ideo'])
        diff_gop = abs(state_ideo - cand_positions.loc[yr,'gop_ideo'])

        if incumbent[yr] == 0:#dem
            regress_data.loc[(yr,state), 'ideo_inc_unweighted'] = cand_positions.loc[yr,'dem_ideo'] 
            regress_data.loc[(yr,state), 'ideo_noninc_unweighted'] = cand_positions.loc[yr,'gop_ideo'] 
            regress_data.loc[(yr,state), 'inc_minus_noninc_new'] = diff_dem - diff_gop
        elif incumbent[yr] == 1:#gop
            regress_data.loc[(yr,state), 'ideo_inc_unweighted'] = cand_positions.loc[yr,'gop_ideo'] 
            regress_data.loc[(yr,state), 'ideo_noninc_unweighted'] = cand_positions.loc[yr,'dem_ideo'] 
            regress_data.loc[(yr,state), 'inc_minus_noninc_new'] = diff_gop - diff_dem 

        if state == 'AK':#workaround for the absence of alaska in the county data
            def eval(x):
                return state_pop.loc[(x,fips), 'Persons: Total'].values[0] / 10**6 * 0.425
            total_votes[state] = linear_interpolate(eval, yr, state_pop['year'])
        else:
            total_votes[state] = states.loc[(yr,fips), 'total_votes'].values[0] / 10**6 * 0.6

        state_weights[f'{state}_weight1'] = df['weight1'] * total_votes[state]
        print('.', end = '')

    natl_weights = state_weights.mean(axis = 'columns')
    quotient_vect = df_yr['weight1'] / natl_weights
    state_weights = state_weights.apply(lambda x: quotient_vect*x)
    adj_natl_weights = state_weights.mean(axis = 'columns')

    sample_convert = pd.DataFrame(df_yr[rake_keys + ['weight1']])
    for key in mapping.keys():#covert all columns that have mappings to census variables
        anes_to_census = mapping[key] 
        sample_convert[key] = df_yr[key].apply(lambda x: anes_to_census[x])
    target_marginal_probs = dict()
    for key in rake_keys:
        target_marginal_probs[key] = sample_convert.groupby(key)['weight1'].agg(pd.Series.sum) / sample_convert['weight1'].sum() 

    state_target_marginals = dict()
    for state in state_postal_codes:
        df = state_df[state]
        df['weight_adj'] = state_weights[f'{state}_weight1']
        adj_share = df.groupby('vote')['weight_adj'].agg(pd.Series.sum) / df['weight_adj'].sum() 
        unadj_share = df.groupby('vote')['weight1'].agg(pd.Series.sum) / df['weight1'].sum() 
        unadj_share = unadj_share.rename(dict(zip(list(unadj_share.keys()), ['unadj_' + x for x in unadj_share.keys()])))
        state_target_marginals[state] = pd.concat([adj_share, unadj_share])

    state_votes = pd.DataFrame(state_target_marginals).transpose()
    state_votes['total_votes'] = pd.Series(total_votes)
    dem_key, gop_key = census_keys['vote'][0], census_keys['vote'][1]
    gop_vote = (sum(state_votes['total_votes'] * state_votes[gop_key])) / state_votes['total_votes'].sum()
    dem_vote = (sum(state_votes['total_votes'] * state_votes[dem_key])) / state_votes['total_votes'].sum()
    total_votes = pd.Series({dem_key: dem_vote, gop_key: gop_vote})
    error = np.mean(abs(total_votes - target_marginal_probs['vote']))

    subs = regress_data[regress_data['year.1'] == yr]
    linear_model = smf.ols('dem_share ~ lean_prev2', data = subs[['dem_share', 'lean_prev2']]).fit()
    model_rsquared = linear_model.rsquared**0.5

    joined = regress_data.loc[yr].join(state_votes[[dem_key,f'unadj_{dem_key}']], how = 'inner')
    corr = np.corrcoef(joined['dem_share'], joined[dem_key])[0,1]
    unadj_corr = np.corrcoef(joined['dem_share'], joined[f'unadj_{dem_key}'])[0,1]
    print(f'{yr} ~ Adjusted Correlation: {corr :<1.3f}, Unadjusted: {unadj_corr :<1.3f}' +
          f', OLS: {model_rsquared :<1.3f}')


