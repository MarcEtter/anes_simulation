from rake import *
from code_to_category import *
from test_regression import * 

OVERWRITE = False

regress_data = pd.read_csv('model_data/regression_data.csv')
regress_data = regress_data.set_index(['year','state'])
#excel workbook containing spliced candidate ideology figures from cohen et al 2016
fundamentals = pd.read_excel('model_data/fundamentals.xlsx')
fundamentals = fundamentals.set_index('Year')

#keys used in ideology paper election model
model_keys = ['inc_share','berry_inc_minus_noninc_citizen','inc_pres','inc_lean_prev','inc_lean_prev2','inc_hshare_prev','inc_hlean_prev','rdi_yr_to_election','inflation_yoy','inc_tenure','inc_home_state','noninc_home_state']
#some additional keys for below analysis
model_keys += ['berry_citizen', 'dem_share', 'spliced_BPHI_inc','spliced_BPHI_noninc','state_evs','inc_evs']
regress_data = regress_data[model_keys]

#code for determining average ideology of state electorates over time
cand_positions = pd.read_csv('model_data/candidate_positions_unweighted.csv')
cand_positions = cand_positions.set_index('year')

#mean and dispersion of cohen presidential candidate ideology measures
all_candidates = pd.concat([regress_data['spliced_BPHI_inc'], regress_data['spliced_BPHI_noninc']])
sd_spliced_BPHI = np.std(all_candidates)
mean_spliced_BPHI = np.mean(all_candidates)

if OVERWRITE:
    for yr in range(1972,2016,4):
        df_yr = pd.read_csv('model_data/simulation_data.csv')
        df_yr = df_yr[df_yr['year'] == yr]
        #rake_keys = list(census_keys.keys())[:-1]
        rake_keys = ['vote']
        state_weights = pd.DataFrame()

        #Find weights of each respondent within state by raking
        state_pop = states[['year','Persons: Total']].dropna()
        state_df = dict()
        total_votes = dict()

        print(f'raked ({yr})')
        for state in state_postal_codes:
            if state == 'OK' and yr == 1996:
                pass
                #large discrepancy observed between berry and raked ideo here
                #print('breakpoint')

            fips = code_to_fips[state]
            state_df[state] = rake_state(yr, fips, df_yr, rake_keys)
            #print(f'raked ({yr},{state})')
            df = state_df[state]
            
            #code for determining average ideology of state electorates over time
            state_ideo = sum(df['ideology'] * df['weight1']) / sum(df['weight1'])
            regress_data.loc[(yr,state), 'ideology'] = state_ideo

            diff_dem = abs(state_ideo - cand_positions.loc[yr,'dem_ideo'])
            diff_gop = abs(state_ideo - cand_positions.loc[yr,'gop_ideo']) 
            #diff_dem = abs(state_ideo - 3.5 - fundamentals.loc[yr, 'spliced.BPHI.dem'])
            #diff_gop = abs(state_ideo - 3.5 - fundamentals.loc[yr, 'spliced.BPHI.rep'])

            regress_data.loc[(yr,state), 'raked_dem_dist'] = diff_dem
            regress_data.loc[(yr,state), 'raked_gop_dist'] = diff_gop

            spliced_BPHI_inc = regress_data.loc[(yr,state), 'spliced_BPHI_inc'] 
            spliced_BPHI_noninc = regress_data.loc[(yr,state), 'spliced_BPHI_noninc']
            berry_ideo = regress_data.loc[(yr,state), 'berry_citizen']

            if incumbent[yr] == 0:#dem
                regress_data.loc[(yr,state), 'ideo_inc_unweighted'] = cand_positions.loc[yr,'dem_ideo'] 
                regress_data.loc[(yr,state), 'ideo_noninc_unweighted'] = cand_positions.loc[yr,'gop_ideo'] 
                regress_data.loc[(yr,state), 'inc_minus_noninc_new'] = diff_dem - diff_gop

                regress_data.loc[(yr,state), 'raked_inc_dist'] = abs(state_ideo - cand_positions.loc[yr,'dem_ideo'])
                regress_data.loc[(yr,state), 'raked_noninc_dist'] = abs(state_ideo - cand_positions.loc[yr,'gop_ideo'])

                regress_data.loc[(yr,state), 'berry_dem_dist'] = abs(berry_ideo - spliced_BPHI_inc)
                regress_data.loc[(yr,state), 'berry_gop_dist'] = abs(berry_ideo - spliced_BPHI_noninc)

            elif incumbent[yr] == 1:#gop
                regress_data.loc[(yr,state), 'ideo_inc_unweighted'] = cand_positions.loc[yr,'gop_ideo'] 
                regress_data.loc[(yr,state), 'ideo_noninc_unweighted'] = cand_positions.loc[yr,'dem_ideo'] 
                regress_data.loc[(yr,state), 'inc_minus_noninc_new'] = diff_gop - diff_dem 

                regress_data.loc[(yr,state), 'raked_inc_dist'] = abs(state_ideo - cand_positions.loc[yr,'gop_ideo'])
                regress_data.loc[(yr,state), 'raked_noninc_dist'] = abs(state_ideo - cand_positions.loc[yr,'dem_ideo'])

                regress_data.loc[(yr,state), 'berry_dem_dist'] = abs(berry_ideo - spliced_BPHI_noninc)
                regress_data.loc[(yr,state), 'berry_gop_dist'] = abs(berry_ideo - spliced_BPHI_inc)

            regress_data.loc[(yr,state), 'berry_inc_dist'] = abs(spliced_BPHI_inc - berry_ideo)
            regress_data.loc[(yr,state), 'berry_noninc_dist'] = abs(spliced_BPHI_noninc - berry_ideo)
            
            if state == 'AK':#workaround for the absence of alaska in the county data
                def eval(x):
                    return state_pop.loc[(x,fips), 'Persons: Total'].values[0] / 10**6 * 0.425
                total_votes[state] = linear_interpolate(eval, yr, state_pop['year'])
            else:
                total_votes[state] = states.loc[(yr,fips), 'total_votes'].values[0] / 10**6 * 0.6

            state_weights[f'{state}_weight1'] = df['weight1'] * total_votes[state]
            #print('.', end = '')
    
    norm_berry_gop_dist = (regress_data['berry_gop_dist'] - np.mean(regress_data['berry_gop_dist'])) 
    norm_berry_gop_dist /= np.std(regress_data['berry_gop_dist'])
    norm_raked_gop_dist = (regress_data['raked_gop_dist'] - np.mean(regress_data['raked_gop_dist'])) 
    norm_raked_gop_dist /= np.std(regress_data['raked_gop_dist'])
    regress_data['resid_gop_dist_berry_rake'] = abs(norm_berry_gop_dist - norm_raked_gop_dist)
    regress_data.to_csv('out/regress_data_state_ideo.csv')

regress_data = pd.read_csv('out/regress_data_state_ideo.csv')

subs = regress_data[regress_data['year'] >= 1972]
subs = subs[subs['year'] <= 2012]
subs = subs[subs['state'] != 'USA']
raked_corr = pd.DataFrame(subs['inc_share']).corrwith(subs['inc_minus_noninc_new'])[0]
berry_corr = pd.DataFrame(subs['inc_share']).corrwith(subs['berry_inc_minus_noninc_citizen'])[0]
#raked_berry_corr = pd.DataFrame(subs['berry_inc_minus_noninc_citizen']).corrwith(subs['inc_minus_noninc_new'])[0]
#raked_corr_inc_dist = pd.DataFrame(subs['inc_share']).corrwith(subs['raked_inc_dist'])[0]
#berry_corr_inc_dist = pd.DataFrame(subs['inc_share']).corrwith(subs['berry_inc_dist'])[0]
#raked_berry_corr_inc_dist = pd.DataFrame(subs['berry_inc_dist']).corrwith(subs['raked_inc_dist'])[0]
#raked_dem_diff = pd.DataFrame(subs['dem_share']).corrwith(subs['raked_dem_dist'])[0]
#raked_gop_diff = pd.DataFrame(1 - subs['dem_share']).corrwith(subs['raked_gop_dist'])[0]
#berry_dem_diff = pd.DataFrame(subs['dem_share']).corrwith(subs['berry_dem_dist'])[0]
#berry_gop_diff = pd.DataFrame(1 - subs['dem_share']).corrwith(subs['berry_gop_dist'])[0]

print(f'Berry diff-diff correlation: {berry_corr :> 1.3f}')
print(f'Raked diff-diff correlation: {raked_corr :> 1.3f}')
#print(f'Berry diff-diff vs. Raked diff-diff correlation: {raked_berry_corr :> 1.3f}')
#print(f'Berry inc dist correlation: {berry_corr_inc_dist :> 1.3f}')
#print(f'Raked inc dist correlation: {raked_corr_inc_dist :> 1.3f}')
#print(f'Berry inc dist vs. Raked inc dist correlation: {raked_berry_corr_inc_dist :> 1.3f}')
#print(f'Raked dem dist vs. dem share correlation: {raked_dem_diff :> 1.3f}')
#print(f'Raked gop dist vs. gop share correlation: {raked_gop_diff :> 1.3f}')
#print(f'Berry dem dist vs. dem share correlation: {berry_dem_diff :> 1.3f}')
#print(f'Berry gop dist vs. gop share correlation: {berry_gop_diff :> 1.3f}')



regression_str = "inc_share ~ berry_inc_minus_noninc_citizen + inc_pres + inc_lean_prev + inc_lean_prev2 + inc_hshare_prev + inc_hlean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure + inc_home_state + noninc_home_state "
model_berry = smf.ols(regression_str, regress_data).fit()
regression_str = "inc_share ~ inc_minus_noninc_new + inc_pres + inc_lean_prev + inc_lean_prev2 + inc_hshare_prev + inc_hlean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure + inc_home_state + noninc_home_state "
model_raked = smf.ols(regression_str, regress_data).fit()

##Compute the accuracy of electoral college predictions for berry and raked ideology

subs[f'berry_pred_inc_share'] = model_berry.predict(regress_data)
subs[f'raked_pred_inc_share'] = model_raked.predict(regress_data)
berry_state_vote = subs['berry_pred_inc_share'].apply(lambda x: 1 if x > 0.5 else 0)
raked_state_vote = subs['raked_pred_inc_share'].apply(lambda x: 1 if x > 0.5 else 0)
true_state_vote = subs['inc_share'].apply(lambda x: 1 if x > 0.5 else 0)
subs[f'berry_state_correct'] = (berry_state_vote == true_state_vote).astype('Int64')
subs[f'raked_state_correct'] = (raked_state_vote == true_state_vote).astype('Int64')
subs[f'berry_pred_evs'] = berry_state_vote * subs['state_evs']
subs[f'raked_pred_evs'] = raked_state_vote * subs['state_evs']

print()
print('{:^50}'.format('-'*50))
print(f'Fundamentals regression R^2 (berry state ideology): {model_berry.rsquared :> 1.3f}')
print(f'Electoral Vote correlation: {np.corrcoef(subs["berry_pred_evs"], subs["inc_evs"])[0,1] :> 1.3f}')
print(f'States correctly called: {np.mean(subs["berry_state_correct"]) :> 1.1%}')
#print(model_berry.summary())
print()
print('{:^50}'.format('-'*50))
print(f'Fundamentals regression R^2 (raked state ideology): {model_raked.rsquared :> 1.3f}')
print(f'Electoral Vote correlation: {np.corrcoef(subs["raked_pred_evs"], subs["inc_evs"])[0,1] :> 1.3f}')
print(f'States correctly called: {np.mean(subs["raked_state_correct"]) :> 1.1%}')
#print(model_raked.summary())

subs.to_csv('out/state_predictions.csv')
