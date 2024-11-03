from code_to_category import *
from test_regression import * 
import plotly.express as px

OVERWRITE = False
WIDTH = 60
yr_range = range(1972,2028,4)

regress_data = pd.read_csv('model_data/regression_data.csv')
regress_data = regress_data.set_index(['year','state'])
regress_data = regress_data.sort_index(level = ['year', 'state'])
#excel workbook containing spliced candidate ideology figures from cohen et al 2016
fundamentals = pd.read_excel('model_data/fundamentals.xlsx')
fundamentals = fundamentals.rename(columns = {'Year':'year'})
fundamentals = fundamentals.set_index('year')
gallup = pd.read_csv('model_data/gallup_2party.csv')
gallup = gallup.set_index('year')

#keys used in ideology paper election model
model_keys = ['inc_share','berry_inc_minus_noninc_citizen','inc_pres','inc_lean_prev','inc_lean_prev2','inc_hshare_prev','inc_hlean_prev','rdi_yr_to_election','inflation_yoy','inc_tenure','inc_home_state','noninc_home_state']
#some additional keys for below analysis
model_keys += ['berry_citizen', 'dem_share', 'spliced_BPHI_inc','spliced_BPHI_noninc','state_evs','inc_evs','inc_candidate','noninc_candidate']
regress_data = regress_data[model_keys]

#code for determining average ideology of state electorates over time
cand_positions = pd.read_csv('model_data/candidate_positions_unweighted.csv')
cand_positions = cand_positions.set_index('year')

#mean and dispersion of cohen presidential candidate ideology measures
all_candidates = pd.concat([regress_data['spliced_BPHI_inc'], regress_data['spliced_BPHI_noninc']])
sd_spliced_BPHI = np.std(all_candidates)
mean_spliced_BPHI = np.mean(all_candidates)

if OVERWRITE:
    states_temp = states.copy()
    states_temp = states_temp.rename(columns = {'us_state': 'state'})
    #since no values for 2024 vote turnout exist, use duplicated 2020 vote turnout
    append_vote_totals = states_temp[states_temp['year'] == 2020]
    append_vote_totals['year'] = 2024
    states_temp = pd.concat([states_temp, append_vote_totals])
    states_temp = states_temp.reset_index(drop = True)
    states_temp = states_temp.set_index(['year','state'])
    #regress_data['fips'] = regress_data['state'].apply(lambda x: code_to_fips[x])
    regress_data = regress_data.join(states_temp['total_votes_two_party'])
    regress_data = regress_data.join(fundamentals['IncumbentPartyMidterm'])
    regress_data = regress_data.join(gallup)

    for yr in yr_range:
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
        for state in state_postal_codes :
            fips = code_to_fips[state]
            state_df[state] = rake_state(yr, fips, df_yr, rake_keys)
            #print(f'raked ({yr},{state})')
            df = state_df[state]
            
            #code for determining average ideology of state electorates over time
            try:
                state_ideo = sum(df['ideology'] * df['weight1']) / sum(df['weight1'])
                regress_data.loc[(yr,state), 'ideology'] = state_ideo
            except:#set ideological indicies to zero if no data exists for the given year
                regress_data.loc[(yr,state), 'ideology'] = 0
                cand_positions.loc[yr,'dem_ideo'] = 0
                cand_positions.loc[yr,'gop_ideo'] = 0

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

                regress_data.loc[(yr,state),'inc_poll'] = regress_data.loc[(yr,state),'dem_poll']
                regress_data.loc[(yr,state),'dem_candidate'] = regress_data.loc[(yr,state),'inc_candidate'][:-5]
                regress_data.loc[(yr,state),'gop_candidate'] = regress_data.loc[(yr,state),'noninc_candidate'][:-5]

            elif incumbent[yr] == 1:#gop
                regress_data.loc[(yr,state), 'ideo_inc_unweighted'] = cand_positions.loc[yr,'gop_ideo'] 
                regress_data.loc[(yr,state), 'ideo_noninc_unweighted'] = cand_positions.loc[yr,'dem_ideo'] 
                regress_data.loc[(yr,state), 'inc_minus_noninc_new'] = diff_gop - diff_dem 

                regress_data.loc[(yr,state), 'raked_inc_dist'] = abs(state_ideo - cand_positions.loc[yr,'gop_ideo'])
                regress_data.loc[(yr,state), 'raked_noninc_dist'] = abs(state_ideo - cand_positions.loc[yr,'dem_ideo'])

                regress_data.loc[(yr,state), 'berry_dem_dist'] = abs(berry_ideo - spliced_BPHI_noninc)
                regress_data.loc[(yr,state), 'berry_gop_dist'] = abs(berry_ideo - spliced_BPHI_inc)

                regress_data.loc[(yr,state),'inc_poll'] = regress_data.loc[(yr,state),'gop_poll']
                regress_data.loc[(yr,state),'dem_candidate'] = regress_data.loc[(yr,state),'noninc_candidate'][:-5]
                regress_data.loc[(yr,state),'gop_candidate'] = regress_data.loc[(yr,state),'inc_candidate'][:-5]

            regress_data.loc[(yr,state), 'berry_inc_dist'] = abs(spliced_BPHI_inc - berry_ideo)
            regress_data.loc[(yr,state), 'berry_noninc_dist'] = abs(spliced_BPHI_noninc - berry_ideo)
            
            if state == 'AK':#workaround for the absence of alaska in the county data
                def eval(x):
                    return state_pop.loc[(x,fips), 'Persons: Total'].values[0] / 10**6 * 0.425
                total_votes[state] = linear_interpolate(eval, yr, state_pop['year'])

                regress_data.loc[(yr,state),'total_votes_two_party'] = int(linear_interpolate(eval, yr, state_pop['year']) * 10**6)
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
regress_data = regress_data[regress_data['state'] != 'USA']

def predict_fn(predict, test_data):
    raked_corr = pd.DataFrame(predict['inc_share']).corrwith(predict['inc_minus_noninc_new'])[0]
    berry_corr = pd.DataFrame(predict['inc_share']).corrwith(predict['berry_inc_minus_noninc_citizen'])[0]

    regression_str = "inc_share ~ berry_inc_minus_noninc_citizen + inc_pres + inc_lean_prev + inc_lean_prev2 + inc_hshare_prev + inc_hlean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure + inc_home_state + noninc_home_state "
    #regression_str = "inc_share ~ inc_poll + inc_pres + inc_lean_prev + inc_lean_prev2 + inc_hshare_prev + inc_hlean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure + inc_home_state + noninc_home_state "
    #regression_str = "inc_share ~ inc_poll + inc_lean_prev2 + inc_lean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure"
    model_berry = smf.ols(regression_str, predict).fit()
    #regression_str = "inc_share ~ inc_minus_noninc_new + inc_pres + inc_lean_prev + inc_lean_prev2 + inc_hshare_prev + inc_hlean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure + inc_home_state + noninc_home_state "
    #regression_str = "inc_share ~ inc_poll + inc_pres + inc_lean_prev2 + inc_lean_prev + rdi_yr_to_election + inflation_yoy + inc_hshare_prev + inc_hlean_prev + inc_tenure + IncumbentPartyMidterm"
    regression_str = "inc_share ~ inc_poll + inc_pres + inc_lean_prev + inc_lean_prev2 + inc_hshare_prev + inc_hlean_prev + rdi_yr_to_election + inflation_yoy + inc_tenure + inc_home_state + noninc_home_state "
    model_raked = smf.ols(regression_str, predict).fit()

    test_data[f'berry_pred_inc_share'] = model_berry.predict(test_data)
    test_data[f'raked_pred_inc_share'] = model_raked.predict(test_data)
    berry_state_vote = test_data['berry_pred_inc_share'].apply(lambda x: 1 if x > 0.5 else 0)
    raked_state_vote = test_data['raked_pred_inc_share'].apply(lambda x: 1 if x > 0.5 else 0)
    true_state_vote = test_data['inc_share'].apply(lambda x: 1 if x > 0.5 else 0)
    test_data[f'berry_state_correct'] = (berry_state_vote == true_state_vote).astype('Int64')
    test_data[f'raked_state_correct'] = (raked_state_vote == true_state_vote).astype('Int64')
    test_data[f'berry_pred_evs_inc'] = berry_state_vote * test_data['state_evs']
    test_data[f'raked_pred_evs_inc'] = raked_state_vote * test_data['state_evs']

    berry_mse = sum(abs((test_data['inc_share'] - test_data['berry_pred_inc_share']).dropna())) / len(test_data)
    raked_mse = sum(abs((test_data['inc_share'] - test_data['raked_pred_inc_share']).dropna())) / len(test_data)

    print(f'{"Berry diff-diff correlation:":<{WIDTH}}' + f'{berry_corr :> 1.3f}')
    print(f'{"Raked diff-diff correlation:":<{WIDTH}}' + f'{raked_corr :> 1.3f}')
    print()
    #print('{:^100}'.format('-'*50))
    print(f'{"Fundamentals regression R^2 (berry state ideology):":<{WIDTH}}' +
          f'{model_berry.rsquared :> 1.3f}')
    #print(f'{"Electoral Vote correlation:":<{WIDTH}}'+
    #      f'{np.corrcoef(test_data["berry_pred_evs"], test_data["inc_evs"])[0,1] :> 1.3f}')
    print(f'{"States correctly called:":<{WIDTH}}'+
          f'{np.mean(test_data["berry_state_correct"]) :> 1.1%}')
    print(f'{"Berry MAE:":<{WIDTH}}' + f'{berry_mse :> 0.2%}')

    #print(model_berry.summary())
    print()
    #print('{:^100}'.format('-'*50))
    print(f'{"Fundamentals regression R^2 (raked state ideology):":<{WIDTH}}' + 
          f'{model_raked.rsquared :> 1.3f}')
    #print(f'{"Electoral Vote correlation:":<{WIDTH}}' + 
    #      f'{np.corrcoef(test_data["raked_pred_evs"], test_data["inc_evs"])[0,1] :> 1.3f}')
    print(f'{"States correctly called:":<{WIDTH}}' + 
          f'{np.mean(test_data["raked_state_correct"]) :> 1.1%}')
    print(f'{"Raked MAE:":<{WIDTH}}' + f'{raked_mse :> 0.2%}')
    #print(model_raked.summary())   

    return test_data

def evaluate():   
    test_data_concat = pd.DataFrame()
    for yr in yr_range:
        print(f'{"-"*int(WIDTH/1.8)}{yr}{"-"*int(WIDTH/1.8)}')
        print()
        predict_data = regress_data[regress_data['year'] != yr]
        test_data = regress_data[regress_data['year'] == yr ]
        current_df = predict_fn(predict_data, test_data)
        current_df['raked_pred_dem'] = current_df['raked_pred_inc_share'].apply(lambda x: x if incumbent[yr] == 0 else 1 - x)
        current_df = current_df.drop(columns = 'berry_inc_minus_noninc_citizen')
        #replace DC vote predicted vote share with 0.85
        current_df['raked_pred_dem'] = current_df['raked_pred_dem'].fillna(0.85)

        current_df['predict_npv_dem'] = sum(current_df['total_votes_two_party'] * current_df['raked_pred_dem']) / sum(current_df['total_votes_two_party'])
        if incumbent[yr] == 0:
            current_df['predict_evs_dem'] = sum(current_df['raked_pred_evs_inc'])
        else:
            current_df['predict_evs_dem'] = sum(current_df['state_evs']) - sum(current_df['raked_pred_evs_inc'])

        test_data_concat = pd.concat([test_data_concat, current_df], axis = 'index')

    test_data_concat['winner'] = test_data_concat['raked_pred_dem'].apply(lambda x: 'dem' if x > 0.5 else 'gop')
    test_data_concat.to_csv('out/state_predictions.csv')

    print(f'{"-"*int(WIDTH/2.5)}Average Accuracy in range {min(yr_range)}-{max(yr_range)}{"-"*int(WIDTH/2.5)}')
    print()
    print(f'{"States correctly called (berry):":<{WIDTH}}'+
          f'{np.mean(test_data_concat["berry_state_correct"]) :> 1.1%}')
    print(f'{"States correctly called (raked):":<{WIDTH}}' + 
          f'{np.mean(test_data["raked_state_correct"]) :> 1.1%}')

    print(f'{"-"*int(WIDTH/2.5)}Overall Accuracy in range {min(yr_range)}-{max(yr_range)}{"-"*int(WIDTH/2.5)}')
    print()
    overall_data = regress_data[regress_data['year'] <= max(yr_range)]
    overall_data = regress_data[regress_data['year'] >= min(yr_range)]
    predict_fn(overall_data, overall_data)

    return test_data_concat

def show(df):
    democrat = pd.Series(df['dem_candidate']).iloc[0]
    republican = pd.Series(df['gop_candidate']).iloc[0]
    dem_npv = pd.Series(df['predict_npv_dem']).iloc[0]
    gop_npv = 1 - dem_npv
    dem_evs = int(pd.Series(df['predict_evs_dem']).iloc[0])
    gop_evs = int(sum(df['state_evs'])) - dem_evs

    fig = px.choropleth(
        df,
        locations="state",
        locationmode="USA-states",
        color = "raked_pred_dem",
        #color="berry_citizen",
        #color_discrete_map= {"dem": "blue",
        #                     "gop": "red"},
        #color_continuous_scale=["red", "blue"],
        #color_continuous_scale=["blue", "red"],
        range_color = [0.48,0.52],
        scope="usa",
        labels={"raked_pred_dem": "Democratic 2-party Vote"},
        labels={"ideology": "ideology"},
        title = f"<span style='font-size: 24px;'>{int(df['year'].mean())} Predicted Vote Share</span>" \
        f"<br><sup><span style='font-size: 16px;'>Popular Vote (D-R): {dem_npv :> 2.1%}-{gop_npv :> 2.1%}</span></sup>" \
        f"<br><sup><span style='font-size: 16px;'>Electoral Vote (D-R): {dem_evs}-{gop_evs}</span></sup>"
    )

    fig.show()

test_data_concat = evaluate()
show(test_data_concat[test_data_concat['year'] == 2024])


