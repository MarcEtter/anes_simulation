import pandas as pd 
import sys
sys.path.insert(1, '../anes_simulation')
from code_to_category import state_name_to_code

opinion = pd.read_csv('state_policy_mood/state_opinion.csv')
opinion['ideology_op'] = 2/100 * (opinion['conservative'] - opinion['liberal']) + 4
opinion['state'] = opinion['statename'].apply(lambda x: state_name_to_code[x])

state_ideo = pd.read_csv('model_data/regression_data.csv').set_index(['year','state'])
opinion = opinion.set_index(['year','state'])
state_ideo = state_ideo.join(opinion['ideology_op'])
state_ideo.to_csv('model_data/regression_data_ideology_op.csv')
