from test_regression import *
import pandas as pd

ELECTION = DEFAULT.copy()
ELECTION['ELECTION_YR'] = 2020
ELECTION['INC_PARTY_CAND_APPROVAL'] = 0.44 #0.44 #Harris approval rating, according to 538
ELECTION['INC_PARTY'] = 0#Democrats 
ELECTION['INC_PARTY_TENURE'] = 4
ELECTION['RDI_YR_TO_ELECTION'] = 50360 / 50151 #RDI PC in Q2 2024 / RDI PC in Q2 2023 https://fred.stlouisfed.org/series/A229RX0Q048SBEA
ELECTION['INFLATION_YOY'] = 1.0289 #CPI in July 2024 / CPI in July 2023
ELECTION['UNEMPLOYMENT'] = 0.042*100 #As of August 2024

if RUN_REGRESSIONS:#if we retabulate the data, save it to file
    sim_df = DEFAULT['DATAFRAME'].copy()
    for key in DEFAULT['DATAFRAME'].keys():
        for party in DEFAULT['PARTIES']:
            if party in key:
                sim_df = sim_df.drop(key) 
    sim_df.to_csv('simulation_data.csv')
else:   
    #Below df is trimmed to only necessary variables, but could leave extra variables in the simulation data csv
    #This would permit the simulation of historical elections where specifying fundamentals above would be redundant
    sim_df = pd.read_csv('simulation_data.csv')

ELECTION['DATAFRAME'] = sim_df
ELECTION['CANDIDATES'] = {
'dem':
    {
    'party': 'dem',
    'code': 0,
    'ideology': 2.1,
    'poll': 0.49},

'gop':
    {
    'party': 'gop',
    'code': 1,
    'ideology': 5.25,
    'poll': 0.46},

'indep':
    {'party': 'indep',
     'code': 2,
    'ideology': 4.25,
    'poll': 0.05}
}
ELECTION['PARTIES'] = [x for x in ELECTION['CANDIDATES'].keys()]
ELECTION['PARTY_CODES'] = dict(zip(ELECTION['PARTIES'], [x['code'] for x in ELECTION['CANDIDATES'].values()]))

sim_df = sim_df[sim_df['year'] == ELECTION['ELECTION_YR']]
sim_df['inc_party_cand_approval'] = ELECTION['INC_PARTY_CAND_APPROVAL']
sim_df['inflation_yoy'] = ELECTION['INFLATION_YOY']
sim_df['inc_party'] = ELECTION['INC_PARTY']
sim_df['inc_party_tenure'] = ELECTION['INC_PARTY_TENURE']
sim_df['rdi_yr_to_election'] = ELECTION['RDI_YR_TO_ELECTION']
sim_df = set_parties(ELECTION)
PARTIES = ELECTION['PARTIES']
model_fundamentals_dict = load_model_fundamentals_multi(REGRESSION_OBJ_PATH, ELECTION['PARTIES'])

for party in PARTIES:
    try:
        if party in ['dem','gop']:
            sim_df[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(sim_df)
        else:
            sim_df[f'{party}_pred_vote_prob'] = 0.5 * (model_fundamentals_dict['dem'].predict(sim_df) + model_fundamentals_dict['gop'].predict(sim_df))
    except KeyError: #party may belong to neither dem nor gop
        sim_df[f'{party}_pred_vote_prob'] = model_fundamentals_dict[party].predict(sim_df)

print(get_vote_shares(sim_df))