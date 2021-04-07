# -*- coding: utf-8 -*-
"""
Functions necessary for fitting SEIR to data
"""

import datetime as dt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy import stats
from SEIR_model_publish_sd_all_contacts import *
import pandas as pd
import os
import datetime
from datetime import timedelta
import multiprocessing as mp
import pickle
import json
from collections import defaultdict
import itertools

from beta_config import *


def calc_residual(fit_var, sim_func, sim_inputs, data):
    # fit 2 params
    beta0_fitting = fit_var[0] * np.ones(N_AGE)
    sd_level = fit_var[1]
    # swap in the new beta
    sim_inputs['beta0'] = beta0_fitting
    sim_inputs['sd_level'] = sd_level
    sim_inputs_final = filter_params(sim_inputs)

    S, E, Pa, Py, Ia, Iy, Ih, R, D, E2Py, E2P, Pa2Ia, Py2Iy, P2I, Iy2Ih, H2D,\
        SchoolCloseTime, SchoolReopenTime = sim_func(**sim_inputs_final)

    fit_compt = Ih.sum(axis=1).sum(axis=1)[range(0,
        TOTAL_TIME * INTERVAL_PER_DAY, INTERVAL_PER_DAY)]

    return fit_compt[:len(data)] - data


def fit_to_data(beta0_guess, sd_level_guess, bnds, sim_func, sim_input, data):

    res = least_squares(
        fun=calc_residual,
        x0=[beta0_guess/100., sd_level_guess],
        bounds=bnds,
        args=(sim_func, sim_input, data)
    )
    soln = res['x'] / np.array([1., 1])

    return soln

def gather_params(growth_scenario, sd_scenario, close_triggers, open_triggers,
                  n_sim, beta, deterministic):

    # a hack to temporarily deal with globals
    from beta_config import SympHratio, SympHratioOverall, HospFratio,\
        run_subgroup
    
    # Extra parameters for subgroups
    if run_subgroup is not None:
        ## Get list of extra parameters specific to subgroup
        # Names of parameter lists
        subgroup_params_names = [k for k in subgroup_param.keys()]
        
        # List of values for each parameter
        subgroup_params_list_values = [subgroup_param[k] for k in \
            subgroup_params_names]
        
        # Every combination of possible parameters (avoids adding nested since
        # the number of parameters needed can theoretically vary)
        subgroup_params_list = list(itertools.\
            product(*subgroup_params_list_values))
        
    else:
        subgroup_params_list = ['']

    # build parameter lists
    scenario_idx = 0
    iter_idx = 0
    param_sets = []
    for g_rate in growth_scenario:
        for sd_level in sd_scenario:
            for c_trigger in close_triggers:
                for r_trigger in open_triggers:
                    for subgroup_params in subgroup_params_list:
                        for sim in range(n_sim):
    
                            Para = SEIR_param_publish.SEIR_get_param(
                                SympHratioOverall,
                                SympHratio,
                                HospFratio,
                                N_AGE,
                                N_RISK,
                                deterministic=deterministic
                            )
                            
                            if run_subgroup is not None:
                                extra_params = {
                                        run_subgroup:[subgroup_params_names,
                                                      subgroup_params]}
                            else:
                                extra_params = None
                            
                                
                            param_sets.append({
                                'full_para': Para,
                                'g_rate': g_rate,
                                'scenario_idx': scenario_idx,
                                'iteration_idx': iter_idx,
                                'sd_level': sd_level,
                                'sd_date': SD_DATE,
                                'metro_pop': MetroPop,
                                'school_calendar': SchoolCalendar,
                                'beta0': beta[g_rate] * np.ones(N_AGE),
                                'phi': Phi,
                                'sigma': Para['sigma'],
                                'gamma': Para['gamma'],
                                'eta': Para['eta'],
                                'mu': Para['mu'],
                                'omega': Para['omega'],
                                'tau': Para['tau'],
                                'nu': Para['nu'],
                                'pi': Para['pi'],
                                'rho': Para['rho'],
                                'n_age': N_AGE,
                                'n_risk': N_RISK,
                                'total_time': TOTAL_TIME,
                                'interval_per_day': INTERVAL_PER_DAY,
                                'shift_week': SHIFT_WEEK,
                                'time_begin': TimeBegin,
                                'time_begin_sim': TIME_BEGIN_SIM,
                                'initial_i': I0,
                                'trigger_type': TRIGGER_TYPE,
                                'close_trigger': c_trigger,
                                'reopen_trigger': r_trigger,
                                'monitor_lag': MONITOR_LAG,
                                'report_rate': REPORT_RATE,
                                'deterministic': deterministic,
                                'print_vals': VERBOSE,
                                'extra_params':extra_params
                            })
                            iter_idx += 1
    
                        # after all N_SIMS parameterized, increment the
                        #   scenario index
                        scenario_idx += 1

    return param_sets


def filter_params(param_dict):

    core_param_keys = [
        'metro_pop','school_calendar','sd_date','sd_level','beta0','phi',
        'sigma', 'gamma','eta','mu','omega','tau','nu','pi','rho','n_age',
        'n_risk','total_time','interval_per_day','shift_week',
        'time_begin','time_begin_sim','initial_i','trigger_type',
        'close_trigger','reopen_trigger','monitor_lag','report_rate',
        'deterministic','print_vals','extra_params']
    
    # grab only the pars needed for the SEIR model
    model_pars = {key: param_dict[key] for key in core_param_keys}

    return model_pars

def single_stochastic_simulation(param_sets):

    outputs = defaultdict(dict)

    model_pars = filter_params(param_sets)

    S, E, Pa, Py, Ia, Iy, Ih, R, D, E2Py, E2P, Pa2Ia, Py2Iy, P2I, Iy2Ih, H2D,\
        SchoolCloseTime, SchoolReopenTime = SEIR_model_publish_w_risk\
        (**model_pars)

    if param_sets['sd_level'] == 0 and param_sets['close_trigger']\
        .split('_')[-1] == '20220101':
        R0 = compute_R0(E2I, INTERVAL_PER_DAY, param_sets['full_para'],
            param_sets['g_rate'])
        R0_baseline = [R0]
    else:
        R0_baseline = []
        
    scenario_metadata = {
            'growth_rate': param_sets['g_rate'],
            'social_distancing': param_sets['sd_level'],
            'school_close_triggers': param_sets['close_trigger'],
            'school_reopen_triggers': param_sets['reopen_trigger'],
        }
    
    # Extra parameters for subgroups

    if run_subgroup is not None:
        extra_params = param_sets['extra_params']
        subgroup = list(extra_params.keys())[0]
        extra_params_details = extra_params[subgroup]
        
        # Get extra parameters names and values
        extra_params_names = extra_params_details[0]
        extra_params_vals = list(extra_params_details[1])
        
        if subgroup == 'Grocery':
            # Contact reduction at grocery store for shoppers due to SD
            g_shopper_sd_idx = extra_params_names.index('g_shopper_sd')
            g_shopper_sd = extra_params_vals[g_shopper_sd_idx]
            
            # Contact reduction at grocery store for workers due to SD
            g_worker_sd_idx = extra_params_names.index('g_worker_sd')
            g_worker_sd = extra_params_vals[g_worker_sd_idx]
            
            scenario_metadata.update({
                    'subgroup':subgroup,
                    'GroceryGShopperSD':g_shopper_sd,
                    'GroceryGWorkerSD':g_worker_sd})
        
        elif subgroup == 'Construction':
            # Social distancing on construction sites
            delta_CW_idx = extra_params_names.index('delta_CW')
            delta_CW = extra_params_vals[delta_CW_idx]
            
            # Proportion of construction workers allowed to work
            prop_CW_idx = extra_params_names.index('prop_CW')
            prop_CW = extra_params_vals[prop_CW_idx]
            
            scenario_metadata.update({
                    'subgroup':subgroup,
                    'ConstructionWorkContactsDelta':delta_CW,
                    'ConstructionPropWorkers':prop_CW})

        # add teacher subgroup
        elif subgroup == 'Teachers':

            # proportion of people in school (t, s, v)
            prop_school_sd_idx = extra_params_names.index('prop_school_sd')
            prop_school_sd = extra_params_vals[prop_school_sd_idx]
            
            # susceptibility
            suscep_param_idx = extra_params_names.index('suscep_param')
            suscep_param = extra_params_vals[suscep_param_idx]
            
            # infectiousness
            infect_param_idx = extra_params_names.index('infect_param')
            infect_param = extra_params_vals[infect_param_idx]

            # put scenario_metadata dict update here
            scenario_metadata.update({
                    'subgroup':subgroup,
                    'prop_school_sd':prop_school_sd,
                    'suscep_param':suscep_param,
                    'infect_param':infect_param
                    })

    else:
        subgroup = 'NoSubgroup'
        scenario_metadata.update({'subgroup':subgroup})
    
    
    outputs['iteration_idx_{}'.format(param_sets['iteration_idx'])] = {
        'scenario_idx': param_sets['scenario_idx'],
        'scenario_metadata': scenario_metadata,
        'scenario_params': param_sets,
        'S': S,
        'E': E,
        'Pa': Pa,
        'Py': Py,
        'Ia': Ia,
        'Iy': Iy,
        'Ih': Ih,
        'R': R,
        'D': D,
        'E2Py': E2Py,
        'E2P': E2P,
        'Pa2Ia': Pa2Ia,
        'Py2Iy': Py2Iy,
        'P2I': P2I,
        'Iy2Ih': Iy2Ih,
        'H2D': H2D,
        'SchoolCloseTime': SchoolCloseTime,
        'SchoolReopenTime': SchoolReopenTime,
        'R0_baseline': R0_baseline
    }

    return outputs

def structure_params(deterministic, beta):
    
    Para = SEIR_param_publish.SEIR_get_param(
        SympHratioOverall,
        SympHratio,
        HospFratio,
        N_AGE,
        N_RISK,
        deterministic
    )

    final_pars = {
        'full_para': Para,
        'g_rate': GROWTH_RATE_LIST,
        'sd_level': SD_LEVEL_LIST_FIT,
        'sd_date': SD_DATE_FIT,
        'metro_pop': MetroPop,
        'school_calendar': SchoolCalendar,
        'beta0': beta * np.ones(N_AGE),
        'phi': Phi,
        'sigma': Para['sigma'],
        'gamma': Para['gamma'],
        'eta': Para['eta'],
        'mu': Para['mu'],
        'omega': Para['omega'],
        'tau': Para['tau'],
        'nu': Para['nu'],
        'pi': Para['pi'],
        'rho': Para['rho'],
        'n_age': N_AGE,
        'n_risk': N_RISK,
        'total_time': TOTAL_TIME,
        'interval_per_day': INTERVAL_PER_DAY,
        'shift_week': SHIFT_WEEK,
        'time_begin': TimeBegin,
        'time_begin_sim': TIME_BEGIN_SIM,
        'initial_i': I0,
        'trigger_type': TRIGGER_TYPE,
        'close_trigger': CLOSE_TRIGGER_LIST,
        'reopen_trigger': REOPEN_TRIGGER_LIST,
        'monitor_lag': MONITOR_LAG,
        'report_rate': REPORT_RATE,
        'deterministic': deterministic,
        'print_vals': VERBOSE
    }
    
    return final_pars

def daily_total_incidence(simulation_out, iter_keys, compartment_key, metric,
                          i_age, group_pop):
    if compartment_key not in set(['E2Iy', 'E2I', 'Iy2Ih', 'H2D', 'E2Py', 'E2P',
        'Pa2Ia', 'Py2Iy', 'P2I']):
        raise ValueError('Daily averages incidence function only valid for incidence compartments E2Iy, E2I, Iy2Ih, or H2D.')

    # Divide by population if computing rate
    if metric == 'count':
        norm = 1.
    elif metric == 'rate':
        norm = group_pop

    daily_hospitalizations = []
    for key in iter_keys:
        if i_age < 0: # total
            dh = simulation_out[key][compartment_key].sum(axis=1).sum(axis=1)\
                .reshape(TOTAL_TIME, INTERVAL_PER_DAY).sum(axis=1) / norm
        else: # specific age groups
            dh = simulation_out[key][compartment_key].sum(axis=2)[:,i_age]\
                .reshape(TOTAL_TIME, INTERVAL_PER_DAY).sum(axis=1) / norm
      
        daily_hospitalizations.append(
            dh.tolist()
        )

    wide = np.vstack(daily_hospitalizations)
    median = np.median(wide, axis=0)
    range_2pt5_97pt5 = np.percentile(wide, [2.5, 97.5], axis=0)
    max = np.amax(wide, axis=0)
    min = np.amin(wide, axis=0)

    summary = pd.DataFrame(
        np.vstack(
            [
                [dt.datetime.strptime(np.str(TIME_BEGIN_SIM), '%Y%m%d') + \
                    dt.timedelta(days=t) for t in range(0, TOTAL_TIME)],
                wide,
                median,
                range_2pt5_97pt5,
                max,
                min
            ]
        ).transpose()
    )

    summary.columns = ['date'] + ['sto_idx_{}'.format(i) for i in \
                      range(len(simulation_out.keys()))] + ['median',
                      'lower_2.5%', 'upper_97.5%', 'max', 'min']

    return summary

def daily_total_cumulative(simulation_out, iter_keys, compartment_key, metric, i_age, group_pop):
    if compartment_key not in set(['E2Iy', 'E2I', 'Iy2Ih', 'H2D', 'E2Py', 'E2P',
        'Pa2Ia', 'Py2Iy', 'P2I']):
        raise ValueError('Daily averages incidence function only valid for incidence compartments E2Iy, E2I, Iy2Ih, or H2D.')

    # Divide by population if computing rate
    if metric == 'count':
        norm = 1.
    elif metric == 'rate':
        norm = group_pop

    daily_hospitalizations = []
    for key in iter_keys:
        if i_age < 0:
            dh = simulation_out[key][compartment_key].sum(axis=1).sum(axis=1)\
                .reshape(TOTAL_TIME, INTERVAL_PER_DAY).sum(axis=1).cumsum() / \
                norm
        else:
            dh = simulation_out[key][compartment_key].sum(axis=2)[:,i_age]\
                .reshape(TOTAL_TIME, INTERVAL_PER_DAY).sum(axis=1).cumsum() / \
                norm
      
        daily_hospitalizations.append(
            dh.tolist()
        )

    wide = np.vstack(daily_hospitalizations)
    median = np.median(wide, axis=0)
    range_2pt5_97pt5 = np.percentile(wide, [2.5, 97.5], axis=0)
    max = np.amax(wide, axis=0)
    min = np.amin(wide, axis=0)

    summary = pd.DataFrame(
        np.vstack(
            [
                [dt.datetime.strptime(np.str(TIME_BEGIN_SIM), '%Y%m%d') + \
                    dt.timedelta(days=t) for t in
                    range(0, TOTAL_TIME)],
                wide,
                median,
                range_2pt5_97pt5,
                max,
                min
            ]
        ).transpose()
    )

    summary.columns = ['date'] + ['sto_idx_{}'.format(i) for i in \
                      range(len(simulation_out.keys()))] + \
                      ['median', 'lower_2.5%', 'upper_97.5%', 'max', 'min']

    return summary

def daily_total_current(simulation_out, iter_keys, compartment_key, metric, i_age, group_pop):

    if compartment_key in set(['E2Iy', 'E2I', 'Iy2Ih', 'H2D', 'E2Py', 'E2P',
        'Pa2Ia', 'Py2Iy', 'P2I']):
        raise ValueError('Daily average total function not valid for incidence compartments E2Iy, E2I, Iy2Ih, or H2D.')

    # Divide by population if computing rate
    if metric == 'count':
        norm = 1.
    elif metric == 'rate':
        norm = group_pop

    daily_hospitalizations = []
    for key in iter_keys:
        if i_age < 0: # total
            dh = simulation_out[key][compartment_key].sum(axis=1)\
                .sum(axis=1)[range(0, TOTAL_TIME * INTERVAL_PER_DAY,
                INTERVAL_PER_DAY)] / norm
        else: # specific age groups
            dh = simulation_out[key][compartment_key].sum(axis=2)\
                [:,i_age][range(0, TOTAL_TIME * INTERVAL_PER_DAY,
                INTERVAL_PER_DAY)] / norm

        daily_hospitalizations.append(
            dh.tolist()
        )

    wide = np.vstack(daily_hospitalizations)
    median = np.median(wide, axis=0)
    range_2pt5_97pt5 = np.percentile(wide, [2.5, 97.5], axis=0)
    max = np.amax(wide, axis=0)
    min = np.amin(wide, axis=0)

    summary = pd.DataFrame(
        np.vstack(
            [
                [dt.datetime.strptime(np.str(TIME_BEGIN_SIM), '%Y%m%d') + \
                dt.timedelta(days=t) for t in range(0, TOTAL_TIME)],
                wide,
                median,
                range_2pt5_97pt5,
                max,
                min
            ]
        ).transpose()
    )

    summary.columns = ['date'] + ['sto_idx_{}'.format(i) for i in \
                      range(len(simulation_out.keys()))] + \
                      ['median', 'lower_2.5%', 'upper_97.5%', 'max', 'min']

    return summary

def sim_only(n_sims, beta):
    sim_params = []
    for i in range(n_sims):
        sim_params.append(structure_params(False, beta=beta))

    recalc = single_stochastic_simulation(sim_params)

    return recalc

def fitting_workflow(n_threads, fitting=True):

    if fitting:

        # load in current data
        case_data = pd.read_csv('hospitalization_data_through_latest.csv')
        data_start_date = dt.datetime.strptime(np.str(case_data['Date'][0]),
            '%Y-%m-%d')
        data_pts = case_data['Hospitalized'].values
        date_begin = dt.datetime.strptime(np.str(TIME_BEGIN_SIM), '%Y%m%d') + \
            dt.timedelta(weeks=SHIFT_WEEK)
        sim_begin_idx = (date_begin - data_start_date).days

        if sim_begin_idx >= 0:
            case_data_values = data_pts[sim_begin_idx: ]
        else:
            case_data_values = np.insert(data_pts, 0, np.zeros(-sim_begin_idx),
                axis=0)

        fit_pars_list = gather_params(
            growth_scenario=GROWTH_RATE_LIST,
            sd_scenario=SD_LEVEL_LIST_FIT,
            close_triggers=CLOSE_TRIGGER_LIST,
            open_triggers=REOPEN_TRIGGER_LIST,
            n_sim=NUM_SIM_FIT,
            beta=BETA0_dict,
            deterministic=True
        )

        if len(fit_pars_list) > 1:
            raise ValueError('{} parameter sets generated for fitting, but only one is allowed. Please check the config file.'.format(len(fit_pars_list)))

        fit_pars = fit_pars_list[0]
        fit_pars.pop('full_para')
        fit_pars.pop('g_rate')

        # pass everything of to the solver
        solution = fit_to_data(
            beta0_guess=0.025 * 100,
            sd_level_guess=0.5,
            bnds=(0, [100, 1]),
            sim_func=SEIR_model_publish_w_risk,
            sim_input=fit_pars,
            data=case_data_values
        )

        use_beta = {GROWTH_RATE_LIST[0]: solution[0]}
        use_sd_level = [solution[1]]
        print('The fitted beta value is {}'.format(use_beta))
        print('The fitted social distancing value is {}'.format(use_sd_level))
        
        # Write fitting results in a text file
        try:
            data_end_date = case_data['Date'].values[-1]
            with open(os.path.join(opts.outputdir, 'FittingResults.txt'), 'w') \
                as fit_file:
                fit_text = data_end_date + '\n' +\
                    'beta: ' + np.str(solution[0]) + '\n' +\
                    'social distancing: ' + np.str(solution[1])
                print(fit_text, file=fit_file)
        except:
            print('Fitting results did not write')

    else:
        use_beta = BETA0_dict
        use_sd_level = SD_LEVEL_LIST

    # generate the parameters needed for this run
    sim_params = gather_params(
        growth_scenario=GROWTH_RATE_LIST,
        sd_scenario=use_sd_level,
        close_triggers=CLOSE_TRIGGER_LIST,
        open_triggers=REOPEN_TRIGGER_LIST,
        n_sim=NUM_SIM,
        beta=use_beta,
        deterministic=True
    )

    pool = mp.Pool(n_threads)
    tasks = [(par,) for par in sim_params]
    results = [pool.apply_async(single_stochastic_simulation, t) for t in tasks]
    pool.close()
    master_dict = defaultdict(dict)
    for r in results:
        sr = r.get()
        # all keys should be unique
        for key, val in sr.items():
            scenario = 'scenario_idx_{}'.format(val['scenario_idx'])
            master_dict[scenario][key] = val

    return master_dict

def main(n_threads, outdir, fit):

    from beta_config import AgeGroupDict, N_AGE, MetroPop
    
    sim_out = fitting_workflow(n_threads=n_threads, fitting=fit)
    
    with open(os.path.join(outdir, 'fitting_and_sim_results.pkl'), 'wb') as out:
        pickle.dump(sim_out, out)

    for scenario, results in sim_out.items():

        # make a home for the scenario-level data
        scenario_path = os.path.join(outdir, scenario)
        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path)

        # save the metadata separately as json for easier access
        grab_key = list(results.keys())[0]
        with open(os.path.join(opts.outputdir, '{}_parameters.txt'\
            .format(scenario)), 'w') as j:
            print(results[grab_key]['scenario_metadata'], file=j)

        # do some routine summary stats and save as CSV
        iteration_keys = results.keys()
        
        incident_dict = {'symp_inf':'Py2Iy',
                         'infections':'P2I',
                         'pre_symp_asymp':'E2P',
                         'hospitalizations':'Iy2Ih',
                         'deaths':'H2D'}
        
        current_dict = {'exposed':'E',
                        'pre_symp':'Py',
                        'pre_asymp':'Pa',
                        'symp_inf':'Iy',
                        'asymp_inf':'Ia',
                        'hospitalizations':'Ih',
                        'recovered':'R'}
        
        for metric in ['count','rate']:
            for i_age in range(-1,N_AGE):
                if i_age < 0:
                    age_group_i = 'Total'
                    group_pop = MetroPop.sum()
                else:
                    age_group_i = AgeGroupDict[N_AGE][i_age]
                    group_pop = MetroPop[i_age,:].sum()
                    
                # Incident and cumulative
                for k,v in incident_dict.items():
                    incident_summary = daily_total_incidence(results,
                        iteration_keys, v, metric, i_age, group_pop)
                    incident_summary.to_csv(
                        os.path.join(scenario_path, '{}_{}_incident_{}_{}.csv'\
                            .format(metric,scenario,k,age_group_i))
                    )
                    
                    cumulative_summary = daily_total_cumulative(results,
                        iteration_keys, v, metric, i_age, group_pop)
                    cumulative_summary.to_csv(
                        os.path.join(scenario_path,
                            '{}_{}_cumulative_{}_{}.csv'\
                            .format(metric,scenario,k,age_group_i))
                    )
                
                # Current
                for k,v in current_dict.items():
                    current_summary = daily_total_current(results,
                        iteration_keys, v, metric, i_age, group_pop)
                    current_summary.to_csv(
                        os.path.join(scenario_path, '{}_{}_current_{}_{}.csv'\
                            .format(metric,scenario,k,age_group_i))
                    )
            
if __name__ == '__main__':
    
    from beta_config import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threads', help='Number of parallel threads to use (default 56 for Frontera).', default=1) # changed: 56 to 4 for Audrey's Mac
    parser.add_argument('-o', '--outputdir', help='Full path to directory for outputs.', default="outputs_test")
    parser.add_argument('-f', '--fit', help='This flag indicates beta should be fitted.', action='store_true')

    opts = parser.parse_args()

    if not os.path.exists(opts.outputdir):
        os.makedirs(opts.outputdir)

    main(int(opts.threads), opts.outputdir, opts.fit)
