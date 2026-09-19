import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

anes_full = pd.read_csv('anes_timeseries_cdf_csv_20260205/anes_timeseries_cdf_csv_20260205.csv')
anes_full = anes_full.rename(columns={
    "VCF0803": "voter_ideo",
    "VCF9096": "gop_ideo",
    "VCF9088": "dem_ideo",
    "VCF0830": "voter_blacks",
    "VCF9092": "gop_blacks",
    "VCF9084": "dem_blacks",
    "VCF0901a": "state_fips",
    "VCF0004": "year"})

anes = anes_full[['year','state_fips','voter_ideo','voter_blacks']]
anes = anes[anes['year'] >= 1972]
anes['voter_ideo'] = pd.to_numeric(anes['voter_ideo'], errors='coerce')
anes['voter_ideo'] = anes['voter_ideo'].apply(lambda x: np.nan if x == 9.0 or x == 0.0 else x)
anes['voter_blacks'] = pd.to_numeric(anes['voter_blacks'], errors='coerce')
anes['voter_blacks'] = anes['voter_blacks'].apply(lambda x: np.nan if x == 9.0 or x == 0.0 else x)
#anes = anes.dropna(subset=['voter_ideo', 'voter_blacks'])

ideo_data = anes[['voter_ideo', 'state_fips', 'year']].dropna().copy()
ideo_data['state_fips'] = ideo_data['state_fips'].astype('string')
ideo_model = OrderedModel(
    ideo_data['voter_ideo'],
    pd.get_dummies(ideo_data[['state_fips', 'year']], columns=['state_fips'], drop_first=True, dtype=float),
    distr='logit')
ideo_mirt = ideo_model.fit(method='bfgs')

blacks_data = anes[['voter_blacks', 'state_fips', 'year']].dropna()
blacks_model = OrderedModel(blacks_data['voter_blacks'],
    pd.get_dummies(blacks_data[['state_fips', 'year']], columns=['state_fips'], drop_first=True, dtype=float),
    distr='logit')
blacks_mirt = blacks_model.fit(method='bfgs')

ideo_mirt.summary()
blacks_mirt.summary()

"""Load the greater US shapefile data in the working directory under greater_us_shapefiles/ and 
create three plots:
1)
2)
3) """


