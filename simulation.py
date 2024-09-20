from test_regression import *
import pandas as pd

election = default.copy()
if RUN_REGRESSIONS:#if we retabulate the data, save it to file
    election['df'] = default['df'].copy()
    for party in default['parties']:
        for key in default['df'].keys():
            if party in key:
                election['df'] = election['df'].drop(key, axis ='columns') 
    election['df'].to_csv('simulation_data.csv')
else:   
    #Below df is trimmed to only necessary variables, but could leave extra variables in the simulation data csv
    #This would permit the simulation of historical elections where specifying fundamentals above would be redundant
    election['df'] = pd.read_csv('simulation_data.csv')

election['year'] = 2020
election['inc_party_cand_approval'] = 0.44*100 #0.44 #Harris approval rating, according to 538
election['inc_party'] = 'dem'#Democrats 
election['inc_party_tenure'] = 4
election['rdi_yr_to_election'] = 50360 / 50151 #RDI PC in Q2 2024 / RDI PC in Q2 2023 https://fred.stlouisfed.org/series/A229RX0Q048SBEA
election['inflation_yoy'] = 1.0289 - 1 #CPI in July 2024 / CPI in July 2023
election['unemployment'] = 0.042*100 #As of August 2024
election['candidates'] = {
'dem':
    {
    'party': 'dem',
    'code': 0,
    'ideology': 2.1,
    'poll': 0.1},

'gop':
    {
    'party': 'gop',
    'code': 1,
    'ideology': 5.25,
    'poll': 0.1},

'indep':
    {'party': 'indep',
     'code': 2,
    'ideology': 4.25,
    'poll': 0.1}
}

election['parties'] = [x for x in election['candidates'].keys()]
election['party_codes'] = dict(zip(election['parties'], [x['code'] for x in election['candidates'].values()]))

sim_df = election['df']
sim_df = sim_df[sim_df['year'] == election['year']]
sim_df['inc_party_cand_approval'] = election['inc_party_cand_approval']
sim_df['inflation_yoy'] = election['inflation_yoy']
sim_df['inc_party'] = election['inc_party']
sim_df['inc_party_tenure'] = election['inc_party_tenure']
sim_df['rdi_yr_to_election'] = election['rdi_yr_to_election']
election['df'] = sim_df
election['df'] = initialize(election)
PARTIES = election['parties']
model_fundamentals_dict = load_model_fundamentals_multi(REGRESSION_OBJ_PATH, election['parties'])
election['df'] = simulate_election(election, model_fundamentals_dict)
print(get_vote_shares(election))

test = {}
i = 1
while i <= 7:
    election['candidates']['indep']['ideology'] = i#3.5 * (3.5**i) / (3.5**i + 1)
    election['candidates']['gop']['ideology'] = 8-i if i <= 4 else 4#3.5 * (3.5**i + 1) / (3.5**i)
    election['df'] = initialize(election)
    PARTIES = election['parties']
    election['df'] = simulate_election(election, model_fundamentals_dict)
    test[f'{i}'] = get_vote_shares(election)
    print(f'Tabulated indep ideology at {i :> 2.2f}...')
    i += 0.1

test = pd.DataFrame(test).transpose()
test.to_csv('test.csv')

sim_df = election['df']
state_shares = dict()
for fips_code in sim_df['fips'].value_counts().keys():
    state_shares[fips_code] = get_vote_shares(election, sim_df[sim_df['fips'] == fips_code])

state_shares = pd.DataFrame(state_shares).transpose()
state_shares['fips'] = state_shares.index
state_shares['state'] = state_shares['fips'].apply(lambda x: state_name[x])
