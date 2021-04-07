# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:46:20 2020

@author: remyp
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import copy
import random
import datetime as dt
#import matplotlib.pyplot as plt
#import seaborn as sns
#import matplotlib.dates as mdates
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

np.set_printoptions(linewidth=115)
pd.set_option('display.width', 115)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.8f}'.format

# Remove simulations that don't take or keep everything
remove_flat_simulations = True

# Whether to separate low and high-risk groups
risk_group_split = False

# Frequency
frequency = 'daily'  # daily   weekly

# Metrics to save results for
# metrics = ['S', 'E', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2Iy', 'E2I', 'Iy2Ih', 'H2D']
metrics = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D', 'E2Py', 'E2P',\
           'Pa2Ia', 'Py2Iy', 'P2I', 'Iy2Ih', 'H2D']

# Quantiles to save (in addition to min, max and median)
quantile_names = ['Q2p5','Q25','Q75','Q97p5'] 
quantiles = [0.025,0.25,0.75,0.975] # nothing: []

# Results folder and
    '''where does the 'win' come from?'''
    
if sys.platform[:3] == 'winnnnn!!!': 
    FOLDER_L = ['C:','Users','remyp','Research','COVID-19','Grocery',\
                'Model','outputs']
#    FOLDER_L = ['Users','audrey','Desktop','Current Projects','COVID',\
#                'teacher_SEIR-master','outputs']
        # Change directory

    '''how does the os.sep.join work?'''
    DIR_FOLDER = os.sep.join(FOLDER_L)
    os.chdir(DIR_FOLDER)
else:
    os.chdir(os.sep.join([os.getcwd(),'outputs']))

## Pickle file to load
# Pickle file should have results for a single subgroup (if any)
pickle_filename = 'fitting_and_sim_results.pkl'


###############################################################################
# simulations = all_iter[:,array_idx,:,:,:]
def summarize(simulations,calc_type, CountRate, i_age, i_risk, group_pop,
              frequency,total_time,interval_per_day,quantiles):
    """ simulations: numpy array. Shape: n simulations x nb steps x n age x n risk
        calc_type: cumul, new, current
        CountRate: string, count or rate (number of people or proportion)
        i_age: age group index. -1 means everything
        i_risk: risk group index. -1 means everything
        group_pop: population in age group. Used to compute rate
        frequency: string. daily or weekly. Results to return
        total_time: days of simulation
        interval_per_day: nb of time steps per day
        quantiles: extra quantiles to compute (does not include min,median, max)
    """
    # Number of simulations
    n_sim = simulations.shape[0]
    
    # Filter for age group and risk group
    if i_risk < 0: # sum over risk groups
        if i_age < 0: # total
            sim_age = simulations.sum(axis=(2,3))
        else: # specific age groups
            sim_age = simulations.sum(axis=3)[:,:,i_age]
    else: # specific risk group
        if i_age < 0: # total
            sim_age = simulations[:,:,:,i_risk].sum(axis=2)
        else: # specific age groups
            sim_age = simulations[:,:,i_age,i_risk]
    
    # Divide by population if computing rate
    if CountRate == 'count':
        norm = 1.
    elif CountRate == 'rate':
        norm = group_pop
        
    # Frequency impacts aggregation
    if frequency == 'daily':
        out_total_time = total_time
        steps_agg = interval_per_day
    elif frequency == 'weekly':
        out_total_time = np.int(total_time/7)
        steps_agg = interval_per_day * 7

    # Get data of interest: new within the period of interest, cumulative
    # or current
    if calc_type == 'new':
        sims = sim_age.reshape(n_sim, out_total_time, steps_agg).sum(axis=2) / norm
    elif calc_type == 'cumul':
        sims = sim_age.reshape(n_sim, out_total_time, steps_agg).sum(axis=2).cumsum(axis=1) / norm
    elif calc_type == 'current':
        # Get data point at the end of the period
        sims = sim_age[:,range(steps_agg-1, (out_total_time+1) * steps_agg -1, steps_agg)] / norm
    
    
    # List of quantiles to measure
    quantiles_full = [0,0.5,1]
    if len(quantiles) > 0:
        quantiles_full.extend(quantiles)
    
    summary = np.quantile(sims,quantiles_full,axis=0)

    return summary
###############################################################################
###############################################################################

'''what's in the pickle file? will the scenario_metadata have info about the teacher subgroup?'''
## Load pickle file
f =  open(pickle_filename, 'rb')
obj = pickle.load(f)
f.close()

# Scenarios: different parameters in each scenario
scenario_list = list(obj.keys())

# Age groups dict
'''does this need to account for employment groups?'''

age_group_dict = {3: ['0-4', '5-17', '18+'], 5: ['0-4', '5-17', '18-49', '50-64', '65+']}
risk_groups = ['LowRisk','HighRisk']

## Parameters to save
measures_out = ['min','median','max'] + quantile_names
cols_out = ['social_distancing','SD_dates','SchoolsClosed',\
            'AgeGroup','RiskGroup','CountRate','Metric','Date'] + measures_out

## Save all results in a single dataframe
df_all = None # pd.DataFrame(columns=cols_out)

for s_i in scenario_list: # s_i = 'scenario_idx_0'
    data_s = obj[s_i] # dictionary containing all the simulations
    iteration_l = list(data_s.keys()) # each simulation has a name 'iteration_idx_X'
    iteration0 = data_s[iteration_l[0]] # iteration0 = data_s['iteration_idx_0']
    
    # Details about all iterations
    scenario_metadata = iteration0['scenario_metadata']
    scenario_params = iteration0['scenario_params']
    SchoolCloseTime = iteration0['SchoolCloseTime']
    SchoolReopenTime = iteration0['SchoolReopenTime']
#    R0_baseline = iteration0['R0_baseline']
    if SchoolCloseTime == 'NA':
        school_dates = 'NoClosure'
    else:
        close_str = dt.datetime.strftime(SchoolCloseTime,'%Y-%m-%d')
        if SchoolReopenTime != 'NA':
            open_str = dt.datetime.strftime(SchoolReopenTime,'%Y-%m-%d')
        else:
            open_str = SchoolReopenTime
        school_dates = '_'.join([close_str,open_str])
    
    # Get specific parameters
    growth_rate = scenario_metadata['growth_rate']
    sd = scenario_metadata['social_distancing']
    SD_dates = scenario_params['sd_date']
    SD_dates_out = '_'.join([dt.datetime.strftime(dt.datetime.strptime(\
        np.str(x), '%Y%m%d'),'%Y-%m-%d') for x in SD_dates])
    
    # Start date
    time_begin_sim = scenario_params['time_begin_sim']
    begin_dt = dt.datetime.strptime(np.str(time_begin_sim), '%Y%m%d')
    
    # Time parameters
    total_time = scenario_params['total_time']
    interval_per_day = scenario_params['interval_per_day']
    
    if frequency == 'daily':
        all_dates = [dt.datetime.strftime(begin_dt + dt.timedelta(days=t),'%Y-%m-%d') \
                     for t in range(total_time)]
    elif frequency == 'weekly':
        n_weeks = np.int(total_time/7)
        all_dates = [dt.datetime.strftime(begin_dt + dt.timedelta(days=6+7*t),'%Y-%m-%d') \
                     for t in range(n_weeks)]
            
    
    # Other parameters
#    beta = scenario_params['beta0'][0]
    n_age = scenario_params['n_age']
    n_risk = scenario_params['n_risk']
    metro_pop = scenario_params['metro_pop']
     
    
    # Subgroup specific details
    if 'subgroup' in scenario_metadata.keys():
        subgroup = scenario_metadata['subgroup'] # to be saved in filename
        age_groups = age_group_dict[n_age]
    else:
        subgroup = ''
        age_groups = age_group_dict[n_age]
    
    # Extra parameters for subgroups
    '''what is this part doing? where did the dict keys come from?'''
    if subgroup == 'Construction':
        age_groups = age_group_dict[5] + ['Construction']
        construction_delta = scenario_metadata['ConstructionWorkContactsDelta']
        construction_prop = scenario_metadata['ConstructionPropWorkers']
        
        if 'construction_delta' not in cols_out:
            cols_out.extend(['construction_delta','construction_prop'])
    
    elif subgroup == 'Grocery':
        age_groups = age_group_dict[5] + ['Grocery']
        g_shopper_sd = scenario_metadata['GroceryGShopperSD']
        g_worker_sd = scenario_metadata['GroceryGWorkerSD']
        
        if 'g_shopper_sd' not in cols_out:
            cols_out.extend(['g_shopper_sd','g_worker_sd'])

    # elif subgroup == 'Teacher':
    
    
    
    ## Loop through simulations and save results
    n_iter = len(iteration_l)
    n_iter_kept = n_iter # to account for simulations that don't take
    
    # Save all data in array, metrics:
    # S, E, Ia, Iy, Ih, R, D, E2Iy, E2I, Iy2Ih, H2D: 11
    n_metrics = len(metrics)
    n_steps = total_time * interval_per_day
    all_iter = np.zeros(shape=(n_iter,n_metrics,n_steps,n_age,n_risk))
    
    for iter_idx in range(n_iter): # iter_idx = 0
        iter_i = iteration_l[iter_idx]
        data_i = data_s[iter_i]
        
        # Check if iteration takes based on final deaths
        total_D = data_i['D'].sum(axis=(1,2))[-1]
        if (total_D < 1) and remove_flat_simulations:
            n_iter_kept = n_iter_kept - 1
            continue # Don't keep data for that iteration
        
        # Save data in array
        for i_m in range(n_metrics):
            m = metrics[i_m]
            all_iter[iter_idx,i_m,:,:,:] = data_i[m]
    
    # Only keep iterations that were filled
    all_iter = all_iter[:n_iter_kept,:,:,:,:]
    
    
    ## Get summary values for every metric
    # Different types of results to calculate
    # incident = ['E2Iy','E2I', 'Iy2Ih', 'H2D']
    # cumulative = ['E2Iy','E2I', 'Iy2Ih']
    # current = ['S', 'E', 'Ia', 'Iy', 'Ih', 'R', 'D']
    incident = ['E2Py', 'E2P','Pa2Ia', 'Py2Iy', 'P2I', 'Iy2Ih', 'H2D']
    cumulative = ['E2Py', 'E2P','Pa2Ia', 'Py2Iy', 'P2I', 'Iy2Ih']
    current = ['S', 'E', 'Pa', 'Py', 'Ia', 'Iy', 'Ih', 'R', 'D']
    
    calculation_types = ['new','cumul','current']
    calculation_types_list = [incident,cumulative,current]
    
    
    ## Get quantiles for all possible metrics / measures
    # Calculate both counts and rates
    
    # Whether we split high and low risk groups
    if risk_group_split:
        n_risk_out = n_risk
    else:
        n_risk_out = 0
    
    # Create dataframe to fill in loop
    n_dates = len(all_dates)
    idx_set = list(range(n_dates))
    # Number iterations in loop: 2 is for count and rate,
    # n risk groups, n age groups and total
    loop_len = 2 * (n_risk_out+1) * (n_age+1) * (sum([len(x) for x in calculation_types_list]))
    
    # Create dataframe that is filled in the loop
    df_loop = pd.DataFrame(np.nan,index=range(loop_len*n_dates),columns=cols_out)
    
    # Summary columns indices
    first_col_idx = cols_out.index('min')
    last_col_idx = first_col_idx + len(measures_out)
    summary_cols_idx = list(range(first_col_idx,last_col_idx))
    
    current_idx = 0
    for CountRate in ['count','rate']:
        # Total and age groups specifically
        for i_risk in range(-1,n_risk_out):
            if i_risk < 0:
                risk_group_i = 'Total'
                risk_pop = metro_pop.sum(axis=1)
            else:
                risk_group_i = risk_groups[i_risk]
                risk_pop = metro_pop[:,i_risk]
            
            for i_age in range(-1,n_age):
                if i_age < 0:
                    age_group_i = 'Total'
                    group_pop = risk_pop.sum()
                else:
                    age_group_i = age_groups[i_age]
                    group_pop = risk_pop[i_age]
                
                # Incident
                for calc_type_idx in range(len(calculation_types)):
                    calc_type = calculation_types[calc_type_idx]
                    compartments_calc = calculation_types_list[calc_type_idx]
                    
                    # Loop through compartments for that type of calc
                    for m_idx in range(len(compartments_calc)): # m_idx = 0
                        metric_i = compartments_calc[m_idx]
                        array_idx = metrics.index(metric_i)
                        simulations = all_iter[:,array_idx,:,:,:]
                        summary = summarize(simulations,calc_type, CountRate, i_age, i_risk, group_pop,
                            frequency,total_time,interval_per_day,quantiles)
                        
                        # Save results
                        cr = [CountRate] * n_dates
                        metric_out = [metric_i + '_' + calc_type] * n_dates
                        
                        rng_i = range(current_idx*n_dates,(current_idx+1)*n_dates)
                        df_loop.loc[rng_i,measures_out] = summary.T
                        df_loop.loc[rng_i,'Date'] = all_dates
                        df_loop.loc[rng_i,'CountRate'] = cr
                        df_loop.loc[rng_i,'Metric'] = metric_out
                        df_loop.loc[rng_i,'AgeGroup'] = age_group_i
                        df_loop.loc[rng_i,'RiskGroup'] = risk_group_i
                        
                        
                        current_idx += 1
    
    # Fill other columns
    df_loop.loc[:,'social_distancing'] = sd
    df_loop.loc[:,'SD_dates'] = SD_dates_out
    df_loop.loc[:,'SchoolsClosed'] = school_dates
    
    # Extra parameters for subgroups
    if subgroup == 'Construction':
        df_loop.loc[:,'construction_delta'] = construction_delta
        df_loop.loc[:,'construction_prop'] = construction_prop
        
    elif subgroup == 'Grocery':
        df_loop.loc[:,'g_shopper_sd'] = g_shopper_sd
        df_loop.loc[:,'g_worker_sd'] = g_worker_sd

    # elif subgroup == 'Teacher':
        # df_loop.loc[:, 'rel_t_contacts'] = ?
        # df_loop.loc[:, 'prop_stu'] = ?
    
    
    # Add to main dataframe
    if df_all is not None:
        df_all = df_all.append(df_loop,ignore_index=True)
    else:
        df_all = df_loop.copy()
            
            
## Export full csv
filename = 'GraphData'
if 'subgroup' in scenario_metadata.keys():
    filename += '_' + subgroup + '.csv'
else:
    filename += '.csv'

df_all.to_csv(filename,index=False)


## If splitting among several files
new_df = df_all.loc[df_all['Metric'].str.contains('new')]
cumul_df = df_all.loc[df_all['Metric'].str.contains('cumul')]
current_df = df_all.loc[df_all['Metric'].str.contains('current')]

# Further split among rates and counts
rates_new_df = new_df.loc[new_df['CountRate'] == 'rate']
count_new_df = new_df.loc[new_df['CountRate'] == 'count']

rates_cumul_df = cumul_df.loc[cumul_df['CountRate'] == 'rate']
count_cumul_df = cumul_df.loc[cumul_df['CountRate'] == 'count']

rates_current_df = current_df.loc[current_df['CountRate'] == 'rate']
count_current_df = current_df.loc[current_df['CountRate'] == 'count']

# Export all to Excel
filename_detail = ['rates_new','count_new','rates_cumul','count_cumul','rates_current','count_current']
df_detail = [rates_new_df,count_new_df,rates_cumul_df,count_cumul_df,rates_current_df,count_current_df]
    
for i_f in range(len(filename_detail)):
    f = filename_detail[i_f]
    df_i = df_detail[i_f]
    filename_i = filename.replace('.csv','_' + f + '.csv')
    df_i.to_csv(filename_i,index=False)



