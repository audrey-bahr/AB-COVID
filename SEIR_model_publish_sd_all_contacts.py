"""
Main file for SEIR model
"""
import school_closure
import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats

np.set_printoptions(linewidth=115)
pd.set_option('display.width', 115)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.8f}'.format

def SEIR_model_publish_w_risk(metro_pop, school_calendar, beta0, phi, sigma,
                              gamma, eta, mu, omega, tau, nu, pi, rho, n_age,
                              n_risk, total_time, interval_per_day, shift_week,
                              time_begin, time_begin_sim, initial_i, sd_date,
                              sd_level, trigger_type, close_trigger,
                              reopen_trigger, monitor_lag, report_rate,
                              deterministic=True, print_vals=True,
                              extra_params=None):
    """
    :param metro_pop: np.array of shape (n_age, n_risk)
    :param school_calendar: np.array of shape(), school calendar from data
    :param beta0: np.array of shape (n_age, ), baseline beta
    :param phi: dict of 4 np.array of shape (n_age, n_age),
        contact matrix of all, school, work, home
    :param sigma: np.array of shape (n_age, ), rate of E to I
    :param gamma: np.array of shape (3, n_age), rate of I to R
    :param eta: old: np.array of shape (n_age, ), rate from I^y to I^H
    :param mu: np.array of shape (n_age, ), rate from I^H to D
    :param omega: np.array of shape (5, n_age), relative infectiousness of I / P
    :param tau: np.array of shape (n_age, ), symptomatic rate of I
    :param nu: np.array of shape (n_risk, n_age), case fatality rate in I^H
    :param pi: np.array of shape (n_risk, n_age), Pr[I^Y to I^H]
    :param rho: np.array of shape (2, n_age), rate P^A/P^I -> I^A/I^Y
    :param n_age: int, number of age groups
    :param n_risk: int, number of risk groups
    :param total_time: int, total length of simulation in (Days)
    :param interval_per_day: int, number of intervals within a day
    :param shift_week: int, shift week !!
    :param time_begin: datetime, time begin
    :param time_begin_sim: int, time to begin simulation
    :param initial_i: np.array of shape(n_age, n_risk), I0
    :param sd_date: list of 2 int, time to start and end social distancing
    :param sd_level: float, % reduction in non-household contacts
    :param trigger_type: str, {'cml', 'current', 'new'}
    :param close_trigger: str, format: type_population_number;
        example: number_all_5 or ratio_school_1 or date__20200315
    :param reopen_trigger: str, format: type_population_number,
        example: monitor_all_75 (75% reduction), no_na_12 (12 weeks)
    :param monitor_lag: int, time lag between surveillance and real time in days
    :param report_rate: float, proportion Y can seen
    :param deterministic: boolean, whether to remove poisson stochasticity
    :param extra_params: dictionary of extra parameters for subgroup if not None
    :return: compt_s, compt_e, compt_ia, compt_ih, compt_ih, compt_r, compt_d,
        compt_e2compt_iy
    """

    date_begin = dt.datetime.strptime(np.str(time_begin_sim), '%Y%m%d') + \
                 dt.timedelta(weeks=shift_week)
    sd_begin_date = dt.datetime.strptime(np.str(sd_date[0]), '%Y%m%d')
    sd_end_date = dt.datetime.strptime(np.str(sd_date[1]), '%Y%m%d')
    sim_begin_idx = (date_begin - time_begin).days
    school_calendar = school_calendar[sim_begin_idx:]

    # Contact matrix for 5 or more age groups, adjusted to time-step
    #if subgroup in ['Grocery', 'Construction', 'Teachers']: # phi = matrices

    phi_all = phi['phi_all'] / interval_per_day
    phi_school = phi['phi_school'] / interval_per_day
    phi_work = phi['phi_work'] / interval_per_day
    phi_home = phi['phi_home'] / interval_per_day
    phi_other = phi['phi_other'] / interval_per_day #phi_all - phi_school - phi_work - phi_home
    
    # Get extra parameters if any
    if extra_params is not None:
        # Subgroup name and parameter names/values
        subgroup = list(extra_params.keys())[0]
        extra_params_details = extra_params[subgroup]
        
        # Get extra parameters names and values
        extra_params_names = extra_params_details[0]
        extra_params_vals = list(extra_params_details[1])
        
        if print_vals:
            print('Subgroup:',subgroup)
            print('Subgroup parameters',extra_params_names,extra_params_vals)
        
        if subgroup == 'Grocery':
            # Grocery store specific contacts and non g_store other contacts
            phi_g_store = phi['phi_g_store'] / interval_per_day
            phi_other_non_gs = phi_other - phi_g_store
            
            # Work contacts split for grocery workers: work on weekends
            phi_work_GW = phi_work.copy() * 0
            phi_work_GW[-1,:] = phi_work[-1,:]
            phi_work[-1,:] = phi_work[-1,:] * 0
            
            # Contact reduction at grocery store for shoppers due to SD
            g_shopper_sd_idx = extra_params_names.index('g_shopper_sd')
            g_shopper_sd = extra_params_vals[g_shopper_sd_idx]
            
            # Contact reduction at grocery store for workers due to SD
            g_worker_sd_idx = extra_params_names.index('g_worker_sd')
            g_worker_sd = extra_params_vals[g_worker_sd_idx]
            
        elif subgroup == 'Construction':
            # Social distancing on construction sites
            delta_CW_idx = extra_params_names.index('delta_CW')
            delta_CW = extra_params_vals[delta_CW_idx]
            
            # Proportion of construction workers allowed to work
            prop_CW_idx = extra_params_names.index('prop_CW')
            prop_CW = extra_params_vals[prop_CW_idx]
            
            # Work contacts split for construction workers
            phi_work_CW = phi_work.copy() * 0
            phi_work_CW[-1,-1] = phi_work[-1,-1]
            phi_work[-1,-1] = 0

        elif subgroup == 'Teachers':
            # get parameters - same as beta config

            # proportion of people in school (t, s, v)
            prop_school_sd_idx = extra_params_names.index('prop_school_sd')
            prop_school_sd = extra_params_vals[prop_school_sd_idx]
            
            # susceptibility
            suscep_param_idx = extra_params_names.index('suscep_param')
            suscep_param = extra_params_vals[suscep_param_idx]
            
            # infectiousness
            infect_param_idx = extra_params_names.index('infect_param')
            infect_param = extra_params_vals[infect_param_idx]
            
            
    if print_vals:
        print('Contact matrices\n\
        All: {}\nSchool: {}\nWork: {}\nHome: {}\nOther places: {}'\
            .format(phi_all * interval_per_day, phi_school * interval_per_day,
                    phi_work * interval_per_day, phi_home * interval_per_day,
                    phi_other * interval_per_day))

    # Rate from symptom onset to hospitalized
    eta = eta / interval_per_day
    if print_vals:
        print('eta', eta)
        print('Duration from symptom onset to hospitalized', 1 / eta / \
            interval_per_day)

    # Symptomatic rate
    if print_vals:
        print('Asymptomatic rate', 1 - tau)

    # Rate from hospitalized to death
    mu = mu / interval_per_day
    if print_vals:
        print('mu', mu)
        print('Duration from hospitalized to death', 1 / mu / interval_per_day)

    # Relative Infectiousness for Ia, Iy, It compartment
    omega_a, omega_y, omega_h, omega_pa, omega_py = omega # CHANGED
    if print_vals:
        print('Relative infectiousness for Ia, Iy, Ih, E is {0} {1} {2} {3}'\
            .format(*omega))

    # Incubation period
    sigma = sigma / interval_per_day
    if print_vals:
        print('sigma', sigma)
        print('Incubation period is {}'.format(1 / sigma / interval_per_day))

    # Recovery rate
    gamma_a, gamma_y, gamma_h = gamma / interval_per_day
    if print_vals:
        print('gamma', gamma_a, gamma_y, gamma_h)
        print('Infectious period for Ia, Iy, Ih is {0} {1} {2}'\
            .format(1 / gamma_a.mean() / interval_per_day,
                    1 / gamma_y.mean() / interval_per_day,
                    1 / gamma_h.mean() / interval_per_day))

    # Rate from pre-symptomatic to symptomatic / asymptomatic
    rho_a, rho_y = rho / interval_per_day # NEW
    if print_vals:
        print('rho', rho_a, rho_y)
        print('Pre-(a)symptomatic period for Pa, Py, is {0} {1}'\
            .format(1 / rho_a.mean() / interval_per_day, 
                    1 / rho_y.mean() / interval_per_day))

    # Case Fatality Rate
    nu_l, nu_h = nu
    if print_vals:
        print('Hosp fatality rate, low risk: {0}. high risk: {1}'.format(*nu))

    # Probability symptomatic go to hospital
    pi_l, pi_h = pi
    if print_vals:
        print('Probability of symptomatic individuals go to hospital', pi)

    # Compartments, axes = (time, age, risk)
    compt_s = np.zeros(shape=(total_time * interval_per_day, n_age, n_risk))
    compt_e, compt_pa, compt_py = compt_s.copy(), compt_s.copy(), compt_s.copy()
    compt_ia, compt_iy = compt_s.copy(), compt_s.copy()
    compt_ih, compt_r, compt_d = compt_s.copy(), compt_s.copy(), compt_s.copy()
    
    # Transitions
    compt_e2compt_p, compt_e2compt_py = compt_s.copy(), compt_s.copy()
    compt_p2compt_i = compt_s.copy() # sum of pa2ia and py2iy
    compt_pa2compt_ia, compt_py2compt_iy = compt_s.copy(), compt_s.copy()
    compt_iy2compt_ih, compt_h2compt_d = compt_s.copy(), compt_s.copy()
    
    # Set initial value for S compartment
    compt_s[0] = metro_pop - initial_i
    compt_py[0] = initial_i

    # Placeholders for
    school_closed = False
    school_reopened = False
    school_close_date = 'NA'
    school_reopen_date = 'NA'

    # Iterate over intervals
    print("sd_all_contacts for loop")
    for t in range(1, total_time * interval_per_day):
  
        days_from_t0 = np.floor((t + 0.1) / interval_per_day)
        t_date = date_begin + dt.timedelta(days=days_from_t0)

        # Use appropriate contact matrix
        # Use different phi values on different days of the week
        if sd_begin_date <= t_date < sd_end_date:
            contact_reduction = sd_level
            sd_active = 1.
        else:
            contact_reduction = 0.
            sd_active = 0.
        
        # Different computations for subgroups
        if extra_params is not None:
            # applying params to reduce contacts
            # contact_reduction - in sd_list, reduces contacts
            if subgroup == 'Grocery':
                if sd_active > 0:
                    GShopper_mult = 1 - g_shopper_sd
                    GWorker_mult = 1 - g_worker_sd
                else:
                    GShopper_mult = 1.
                    GWorker_mult = 1.
                    
                phi_weekday = (1 - contact_reduction) * \
                    (phi_home + phi_work + phi_school + phi_other_non_gs) + \
                    phi_work_GW * GWorker_mult + phi_g_store * GShopper_mult
                phi_weekend = (1 - contact_reduction) * (phi_home + \
                    phi_other_non_gs) + phi_work_GW * GWorker_mult + \
                    phi_g_store * GShopper_mult

            elif subgroup == 'Construction':
                # Contacts only adjusted when social distancing in place
                if sd_active > 0:
                    CW_multiplier = delta_CW * prop_CW
                else:
                    CW_multiplier = 1
                
                # Construction workers' work contacts not impacted the same
                # when social distancing inplace
                phi_weekday = (1 - contact_reduction) * \
                    (phi_home + phi_work + phi_school + phi_other) + \
                    phi_work_CW  * CW_multiplier
                phi_weekend = (1 - contact_reduction) * (phi_home + phi_other)

            elif subgroup == 'Teachers':
                # consistent reduction for groups
                # contact_reduction = ?
                if sd_active > 0:
                    school_contact_reduction = prop_school_sd
                else:
                    school_contact_reduction = 0
                    
                phi_weekday = (1 - contact_reduction) * \
                    (phi_home + phi_work + phi_other) + \
                    phi_school * (1 - school_contact_reduction)
                phi_weekend = (1 - contact_reduction) * (phi_home + phi_other)

                    
        else:
            # No subgroup
            phi_weekday = (1 - contact_reduction) * phi_all
            phi_weekend = (1 - contact_reduction) * (phi_all - phi_school - \
                phi_work)
        
        phi_weekday_holiday = phi_weekend
        phi_weekday_long_break = phi_weekday - (1 - contact_reduction) * \
            phi_school
        if subgroup == 'Teachers':
            phi_weekday_long_break = phi_weekday - (1 - school_contact_reduction) * \
            phi_school
            
        phi_open = [phi_weekday, phi_weekend, phi_weekday_holiday,
            phi_weekday_long_break]
        phi_close = [phi_weekday - (1 - contact_reduction) * phi_school,
            phi_weekend, phi_weekday_holiday, phi_weekday_long_break]


        # 1-weekday, 2-weekend, 3-weekday holiday, 4-weekday long break
        calendar_code = int(school_calendar[int(days_from_t0)])
        if school_closed == school_reopened:
            phi = phi_open[calendar_code - 1]
        else:
            phi = phi_close[calendar_code - 1]

        temp_s = np.zeros(shape=(n_age, n_risk))
        temp_e = np.zeros(shape=(n_age, n_risk))
        temp_e2py = np.zeros(shape=(n_age, n_risk))
        temp_e2p = np.zeros(shape=(n_age, n_risk))
        temp_pa = np.zeros(shape=(n_age, n_risk))
        temp_py = np.zeros(shape=(n_age, n_risk))
        temp_pa2ia = np.zeros(shape=(n_age, n_risk))
        temp_py2iy = np.zeros(shape=(n_age, n_risk))
        temp_p2i = np.zeros(shape=(n_age, n_risk))
        temp_ia = np.zeros(shape=(n_age, n_risk))
        temp_iy = np.zeros(shape=(n_age, n_risk))
        temp_ih = np.zeros(shape=(n_age, n_risk))
        temp_r = np.zeros(shape=(n_age, n_risk))
        temp_d = np.zeros(shape=(n_age, n_risk))
        temp_iy2ih = np.zeros(shape=(n_age, n_risk))
        temp_h2d = np.zeros(shape=(n_age, n_risk))

        ## within nodes
        # for each age group
        for a in range(n_age):
      
            # for each risk group
            for r in range(n_risk):
           
                rate_s2e = 0.

                if r == 0:  # p0 is low-risk group, 1 is high risk group
                    temp_nu = nu_l
                    temp_pi = pi_l
                else:
                    temp_nu = nu_h
                    temp_pi = pi_h

                # Calculate infection force (F)                         
                for a2 in range(n_age):
                    for r2 in range(n_risk):
                        # multiply omega_y by number - relative infectiousness per age group, each age group has a specific one
                        rate_s2e += suscep_param[a] * infect_param[a2] * beta0[a2] * phi[a, a2] * \
                            compt_s[t - 1, a, r] * \
                            (omega_a[a2] * compt_ia[t - 1, a2, r2] + \
                            omega_y[a2] * compt_iy[t - 1, a2, r2] + \
                            omega_pa[a2] * compt_pa[t - 1, a2, r2] + \
                            omega_py[a2] * compt_py[t - 1, a2, r2]) + \
                            np.sum(metro_pop[a2])
                                                
                if np.isnan(rate_s2e):
                    rate_s2e = 0

                # Rate change of each compartment
                # (besides S -> E calculated above)
                rate_e2p = sigma[a] * compt_e[t - 1, a, r]
                rate_pa2ia = rho_a[a] * compt_pa[t - 1, a, r]
                rate_py2iy = rho_y[a] * compt_py[t - 1, a, r]
                rate_ia2r = gamma_a[a] * compt_ia[t - 1, a, r]
                rate_iy2r = (1 - temp_pi[a]) * gamma_y[a] * \
                    compt_iy[t - 1, a, r]
                rate_ih2r = (1 - temp_nu[a]) * gamma_h[a] * \
                    compt_ih[t - 1, a, r]
                rate_iy2ih = temp_pi[a] * eta[a] * compt_iy[t - 1, a, r]
                rate_ih2d = temp_nu[a] * mu[a] * compt_ih[t - 1, a, r]

                # Stochastic rates
                if not deterministic:
                    rate_s2e = np.random.poisson(rate_s2e)
                if np.isinf(rate_s2e):
                    rate_s2e = 0

                if not deterministic:
                    rate_e2p = np.random.poisson(rate_e2p)
                if np.isinf(rate_e2p):
                    rate_e2p = 0
                
                if not deterministic:
                    rate_pa2ia = np.random.poisson(rate_pa2ia)
                if np.isinf(rate_pa2ia):
                    rate_pa2ia = 0
                
                if not deterministic:
                    rate_py2iy = np.random.poisson(rate_py2iy)
                if np.isinf(rate_py2iy):
                    rate_py2iy = 0 # NEW

                if not deterministic:
                    rate_ia2r = np.random.poisson(rate_ia2r)
                if np.isinf(rate_ia2r):
                    rate_ia2r = 0

                if not deterministic:
                    rate_iy2r = np.random.poisson(rate_iy2r)
                if np.isinf(rate_iy2r):
                    rate_iy2r = 0

                if not deterministic:
                    rate_ih2r = np.random.poisson(rate_ih2r)
                if np.isinf(rate_ih2r):
                    rate_ih2r = 0

                if not deterministic:
                    rate_iy2ih = np.random.poisson(rate_iy2ih)
                if np.isinf(rate_iy2ih):
                    rate_iy2ih = 0

                if not (deterministic):
                    rate_ih2d = np.random.poisson(rate_ih2d)
                if np.isinf(rate_ih2d):
                    rate_ih2d = 0

                # In the below block, calculate values + deltas of each category
                # in SEIR, for each age-risk category, at this timepoint

                d_s = -rate_s2e
                temp_s[a, r] = compt_s[t - 1, a, r] + d_s
                if temp_s[a, r] < 0:
                    rate_s2e = compt_s[t - 1, a, r]
                    temp_s[a, r] = 0

                d_e = rate_s2e - rate_e2p
                temp_e[a, r] = compt_e[t - 1, a, r] + d_e
                if temp_e[a, r] < 0:
                    rate_e2p = compt_e[t - 1, a, r] + rate_s2e
                    temp_e[a, r] = 0

                temp_e2p[a, r] = rate_e2p
                temp_e2py[a, r] = tau[a] * rate_e2p
                if temp_e2py[a, r] < 0:
                    rate_e2p = 0
                    temp_e2p[a, r] = 0
                    temp_e2py[a, r] = 0

                d_pa = (1 - tau[a]) * rate_e2p - rate_pa2ia
                temp_pa[a, r] = compt_pa[t - 1, a, r] + d_pa
                temp_pa2ia[a, r] = rate_pa2ia
                if temp_pa[a, r] < 0:
                    rate_pa2ia = compt_pa[t - 1, a, r] + (1 - tau[a]) * rate_e2p
                    temp_pa[a, r] = 0
                    temp_pa2ia[a, r] = rate_pa2ia
                
                d_py = tau[a] * rate_e2p - rate_py2iy
                temp_py[a, r] = compt_py[t - 1, a, r] + d_py
                temp_py2iy[a, r] = rate_py2iy
                if temp_py[a, r] < 0:
                    rate_py2iy = compt_py[t - 1, a, r] + tau[a] * rate_e2p
                    temp_py[a, r] = 0
                    temp_py2iy[a, r] = rate_py2iy
                
                d_ia = rate_pa2ia - rate_ia2r
                temp_ia[a, r] = compt_ia[t - 1, a, r] + d_ia
                if temp_ia[a, r] < 0:
                    rate_ia2r = compt_ia[t - 1, a, r] + rate_pa2ia
                    temp_ia[a, r] = 0
                
                d_iy = rate_py2iy - rate_iy2r - rate_iy2ih
                temp_iy[a, r] = compt_iy[t - 1, a, r] + d_iy
                if temp_iy[a, r] < 0:
                    rate_iy2r = (compt_iy[t - 1, a, r] + rate_py2iy) * \
                        rate_iy2r / (rate_iy2r + rate_iy2ih)
                    rate_iy2ih = compt_iy[t - 1, a, r] + rate_py2iy - rate_iy2r
                    temp_iy[a, r] = 0
                
                temp_iy2ih[a, r] = rate_iy2ih
                if temp_iy2ih[a, r] < 0:
                    temp_iy2ih[a, r] = 0

                d_ih = rate_iy2ih - rate_ih2r - rate_ih2d
                temp_ih[a, r] = compt_ih[t - 1, a, r] + d_ih
                if temp_ih[a, r] < 0:
                    rate_ih2r = (compt_ih[t - 1, a, r] + rate_iy2ih) * \
                        rate_ih2r / (rate_ih2r + rate_ih2d)
                    rate_ih2d = compt_ih[t - 1, a, r] + rate_iy2ih - rate_ih2r
                    temp_ih[a, r] = 0

                d_r = rate_ia2r + rate_iy2r + rate_ih2r
                temp_r[a, r] = compt_r[t - 1, a, r] + d_r

                d_d = rate_ih2d
                temp_h2d[a, r] = rate_ih2d
                temp_d[a, r] = compt_d[t - 1, a, r] + d_d

        # We are now done calculating compartment values for each
        # age-risk category
        # Copy this vector array as a slice on time axis
        compt_s[t] = temp_s
        compt_e[t] = temp_e
        compt_pa[t] = temp_pa
        compt_py[t] = temp_py
        compt_ia[t] = temp_ia
        compt_iy[t] = temp_iy
        compt_ih[t] = temp_ih
        compt_r[t] = temp_r
        compt_d[t] = temp_d
        
        compt_e2compt_p[t] = temp_e2p
        compt_e2compt_py[t] = temp_e2py
        compt_pa2compt_ia[t] = temp_pa2ia
        compt_py2compt_iy[t] = temp_py2iy
        compt_p2compt_i[t] = temp_pa2ia + temp_py2iy
        compt_iy2compt_ih[t] = temp_iy2ih
        compt_h2compt_d[t] = temp_h2d
        
        # Check if school closure is triggered
        t_surveillance = np.maximum(t - monitor_lag * interval_per_day, 0)
        # Current number of infected
        current_iy = compt_iy[t_surveillance]
        new_iy = compt_py2compt_iy[t_surveillance] # NEW
        cml_iy = np.sum(compt_py2compt_iy[:(t_surveillance + 1)], axis=0)
        trigger_type_dict = {'cml': cml_iy, 'current': current_iy, 'new': new_iy}
        trigger_iy = trigger_type_dict[trigger_type.lower()]

        if not school_closed:
            school_closed = school_closure.school_close(close_trigger, t_date,
                trigger_iy, metro_pop)
            if school_closed:
                school_close_time = t
                school_close_date = t_date
                school_close_iy = trigger_iy
        else:
            if not school_reopened:
                school_reopened = school_closure.school_reopen(reopen_trigger,
                    school_close_iy, trigger_iy, school_close_time, t, t_date,
                    interval_per_day)
                if school_reopened:
                    school_reopen_date = t_date

    return compt_s, compt_e, compt_pa, compt_py, compt_ia, compt_iy, compt_ih, \
           compt_r, compt_d, compt_e2compt_py,compt_e2compt_p, \
           compt_pa2compt_ia, compt_py2compt_iy, compt_p2compt_i, \
           compt_iy2compt_ih, compt_h2compt_d, school_close_date, \
           school_reopen_date

def compute_R0(compt_p2compt_i, interval_per_day, para, growth_rate):
    """
    :param: np.array containing new symptomatic cases in each time step
    :param: interval_per_day: int, number of time steps per day
    :param: para: dict, parameter dictionary
    :return: a single number as estimate of R0
    """
    # Get total cases (summed over risk and age groups)
    cases_ts = compt_p2compt_i.sum(axis=2).sum(axis=1)

    # Get generation time implied by doubling time and growth rate
    R0_obj = para['r0']  # target R0
    double_time = para['double_time'][growth_rate]
    gen_time = double_time * (R0_obj - 1) / (np.log(2))

    # Find peak of new cases
    max_idx = np.where(cases_ts == cases_ts.max())[0][0]

    # Remove last full day of data (plus part of day if any)
    cutoff = max_idx - np.mod(max_idx, interval_per_day) - interval_per_day
    growing_cases = cases_ts[:cutoff]

    # Get number of days in time series, aggregate per day
    nb_days = np.int(cutoff / interval_per_day)
    cases_daily = growing_cases.reshape(nb_days, interval_per_day).sum(axis=1)

    # Compute growth rate (Get rid of starting zeros for log)
    nb_zeros = len(np.where(cases_daily == 0)[0])
    if nb_zeros > 0:
        min_pos_day = np.where(cases_daily == 0)[0][-1] + 1
    else:
        min_pos_day = 0

    if min_pos_day < len(cases_daily):
        time = list(range(min_pos_day, nb_days))
        log_cases = np.log(cases_daily[min_pos_day:])
        growth_rate, y0, r_val, p_val, std_err = stats.linregress(time,
            log_cases)

        # Get estimate R0
        R0 = gen_time * growth_rate + 1
    else:
        R0 = 0

    return R0
