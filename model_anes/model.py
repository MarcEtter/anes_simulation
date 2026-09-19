

def rake(sample, target_marginal_probs, rake_keys):
    sample_marginal_probs = {}
    #creates dict of dataframes, each containing marginal probabilities of categories for one categorical variable
    scalars = dict()
    max_iter = 10
    i = 0
    converge = dict(zip(rake_keys, [False] * len(rake_keys)))
    converge_thresh = 0.9975
    corr = 0

    #convert sample categories to target categories
    sample_convert = sample.copy() #pd.DataFrame(sample[rake_keys + ['weight1']])
    for key in mapping.keys():#covert all columns that have mappings to census variables
        anes_to_census = mapping[key] 
        sample_convert[key] = sample[key].apply(lambda x: anes_to_census[x])

    while i < max_iter and not all(converge.values()):
        prob_table = pd.DataFrame()
        for key in rake_keys:
            #need to update sample_marginal_probs after each iteration
            sample_marginal_probs[key] = sample_convert.groupby(key)['weight1'].agg(pd.Series.sum) / sample_convert['weight1'].sum() 
            target = pd.Series(target_marginal_probs[key].loc[0])
            joined = pd.concat([target, sample_marginal_probs[key]], axis = 'columns').fillna(1)
            scalars_new = dict(joined.iloc[:,0] / joined.iloc[:,1])

            #create dataframe to compute correlation between sample and target marginals
            sample = pd.Series(sample_marginal_probs[key]) 
            target = pd.Series(target_marginal_probs[key].loc[0])
            prob_table = pd.concat([prob_table, pd.concat([sample, target], axis = 'columns')], axis = 'index')
            
            if i > 0 and corr > converge_thresh:
                converge[key] = True
            else:
                scalars[key] = scalars_new

            #convert anes category to census category and get corresponding scalar
            scalar_vect = sample_convert[key].apply(lambda x: scalars_new[x])
            sample_convert['weight1'] = sample_convert['weight1'] * scalar_vect

        corr = np.corrcoef(prob_table.iloc[:,0].fillna(0), prob_table.iloc[:,1].fillna(0))[1,0]
        i+=1

    if PRINT_RAKING:
        print(f'Converged after {i} iterations. \n' + 
            f'Marginal probability correlation: {corr :<1.4f}')

    return sample_convert