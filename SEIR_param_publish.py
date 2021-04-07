"""
SEIR model: get input data from Excel files
"""
import numpy as np
import pandas as pd
import datetime as dt

R0 = 2.2
DOUBLE_TIME = {'high': 4., 'low': 7.2, 'fit': 2.797335}

T_EXPOSED_PARA = [1.9, 2.9, 3.9]
T_EXPOSED_DIST = lambda x: np.random.triangular(*x)

T_Y_TO_R_PARA = np.array([3., 4., 5.])
T_Y_TO_R_DIST = lambda x: np.random.triangular(*x)

ASYMP_RATE = 0.43 
ASYMP_RELATIVE_INFECT = 2/3

PROP_TRANS_IN_P = 0.44
T_PA_TO_IA = 2.3
T_PY_TO_IY = 2.3

T_ONSET_TO_H = 5.9

T_H_TO_R = 7.3
T_H_TO_R_65 = 9.9

T_H_TO_D = 17.8
T_H_TO_D_65 = 10.6

H_RELATIVE_RISK_IN_HIGH = 10
D_RELATIVE_RISK_IN_HIGH = 10

# All in %
#'18-49': 16.5298, # split - 18-25, 26-49
HIGH_RISK_RATIO = {
    '0-4': 8.3651,
    '5-17': 14.3375,
    '18-25': 15.3698,
    '26-49': 20.00,
    '50-64': 32.02,
    '65+': 47.10
}

H_FATALITY_RATIO = {
    '0-9': 0.,
    '10-19': 0.2,
    '20-29': 0.2,
    '30-39': 0.2,
    '40-49': 0.4,
    '50-59': 1.3,
    '60-69': 3.6,
    '70-79': 8,
    '80+': 14.8
}

INFECTION_FATALITY_RATIO = {
    '0-9': 0.0016,
    '10-19': 0.007,
    '20-29': 0.031,
    '30-39': 0.084,
    '40-49': 0.16,
    '50-59': 0.6,
    '60-69': 1.9,
    '70-79': 4.3,
    '80+': 7.8
}

OVERALL_H_RATIO = {
    '0-9': 0.04,
    '10-19': 0.04,
    '20-29': 1.1,
    '30-39': 3.4,
    '40-49': 4.3,
    '50-59': 8.2,
    '60-69': 11.8,
    '70-79': 16.6,
    '80+': 18.4
}

# new dictionaries - relative susceptibility, # infectiousness - isn't that already in beta_config

###############################################################################
def SEIR_get_data(data_folder, city, n_age, n_risk, run_subgroup=None): # where is this called?
    """ Gets input data from Excel files
    :param data_folder: str, path of Excel files
    :param city: str, name of city simulated
    :param n_age: int, number of age groups
    :param n_risk: int, number of risk groups
    run_subgroup: grocery workers, construction workers or None
    """

    age_group_dict = {
        3: ['0-4', '5-17', '18+'],
        5: ['0-4', '5-17', '18-49', '50-64', '65+']
    }

    # data_folder = Austin Data?
    ### Input data
    ## File names
    us_population_filename = 'US_pop_UN.csv' # if run_subgroup != 'Teachers'
    population_filename = ' Population - X age groups.csv'
    population_filename_dict = {}
    for key in age_group_dict.keys():
        population_filename_dict[key] = city + population_filename.replace('X',
            str(key))
    
    if run_subgroup is not None:
        if run_subgroup in ['Grocery','Construction']: 
            age_group_dict.update({6:['0-4', '5-17', '18-49', '50-64', '65+', 
                run_subgroup]})
    
        if run_subgroup == 'Grocery':
            pop_path = run_subgroup + '/AustinMSA Population - 5 age groups and'
            pop_path += ' grocery workers.csv'
            population_filename_dict[6] = pop_path
        
        if run_subgroup == 'Construction':
            pop_path = run_subgroup + '/AustinMSA Population - 5 age groups and'
            pop_path += 'construction workers.csv'
            population_filename_dict[6] = pop_path

        if run_subgroup == 'Teachers': # update dictionary as above for N_AGE = 14 groups
            us_population_filename = 'US_pop_UN_6.csv'
            age_group_dict.update({14:["0-4","5-17","18-25 T","18-25 S","18-25 X",
                                       "26-49 T","26-49 S","26-49 X",
                                       "50-64 T","50-64 S","50-64 X",
                                       "65+ T", "65+ S","65+ X"]})
            
            pop_path = run_subgroup + '/Austin Population - 6 age groups, school employment.csv' # population 
            population_filename_dict[14] = pop_path
            

    school_calendar_filename = city + ' School Calendar.csv'

# not used for teacher calcs
    contact_matrix_all_filename_dict = {
        5: 'ContactMatrixAll_5AgeGroups.csv',
        3: 'ContactMatrixAll_3AgeGroups.csv' 
    }
    contact_matrix_school_filename_dict = {
        5: 'ContactMatrixSchool_5AgeGroups.csv',
        3: 'ContactMatrixSchool_3AgeGroups.csv'
        }
    contact_matrix_work_filename_dict = {
        5: 'ContactMatrixWork_5AgeGroups.csv',
        3: 'ContactMatrixWork_3AgeGroups.csv'
        }
    contact_matrix_home_filename_dict = {
        5: 'ContactMatrixHome_5AgeGroups.csv',
        3: 'ContactMatrixHome_3AgeGroups.csv'
    }

    ## Load data
    # Population in US
    df_US = pd.read_csv(data_folder + us_population_filename, index_col=False)
    GroupPaperPop = df_US.groupby('GroupPaper')['Value'].sum()\
        .reset_index(name='GroupPaperPop')
    GroupCOVIDPop = df_US.groupby('GroupCOVID')['Value'].sum()\
        .reset_index(name='GroupCOVIDPop')
    df_US = pd.merge(df_US, GroupPaperPop)
    df_US = pd.merge(df_US, GroupCOVIDPop)

    # Calculate age/risk group specific symptomatic hospitalization ratio
    df_US['Overall_H_Ratio'] = df_US['GroupPaper'].map(OVERALL_H_RATIO) / 100.
    df_US['YHR_paper'] = df_US['Overall_H_Ratio'] / (1 - ASYMP_RATE)
    df_US['YHN_1yr'] = df_US['YHR_paper'] * df_US['Value']
    GroupCOVID_YHN = df_US.groupby('GroupCOVID')['YHN_1yr'].sum()\
        .reset_index(name='GroupCOVID_YHN')
    df_US = pd.merge(df_US, GroupCOVID_YHN)
    df_US['YHR'] = df_US['GroupCOVID_YHN'] / df_US['GroupCOVIDPop']
    df_US['GroupCOVIDHighRiskRatio'] = df_US['GroupCOVID']\
        .map(HIGH_RISK_RATIO) / 100.
    df_US['YHR_low'] = df_US['YHR'] /(1 - df_US['GroupCOVIDHighRiskRatio'] + \
        H_RELATIVE_RISK_IN_HIGH * df_US['GroupCOVIDHighRiskRatio'])
    df_US['YHR_high'] = H_RELATIVE_RISK_IN_HIGH * df_US['YHR_low']

    # Calculate age specific and risk group specific hospitalized fatality ratio
    df_US['I_Fatality_Ratio'] = df_US['GroupPaper']\
        .map(INFECTION_FATALITY_RATIO) / 100.
    df_US['YFN_1yr'] = df_US['I_Fatality_Ratio'] * df_US['Value'] / \
        (1 - ASYMP_RATE)
    GroupCOVID_YFN = df_US.groupby('GroupCOVID')['YFN_1yr'].sum()\
        .reset_index(name='GroupCOVID_YFN')
    df_US = pd.merge(df_US, GroupCOVID_YFN)
    df_US['YFR'] = df_US['GroupCOVID_YFN'] / df_US['GroupCOVIDPop']
    df_US['YFR_low'] = df_US['YFR'] / (1 - df_US['GroupCOVIDHighRiskRatio'] + \
        D_RELATIVE_RISK_IN_HIGH * df_US['GroupCOVIDHighRiskRatio'])
    df_US['YFR_high'] = D_RELATIVE_RISK_IN_HIGH * df_US['YFR_low']
    df_US['HFR'] = df_US['YFR'] / df_US['YHR']
    df_US['HFR_low'] = df_US['YFR_low'] / df_US['YHR_low']
    df_US['HFR_high'] = df_US['YFR_high'] / df_US['YHR_high']

    df_US_dict = df_US[['GroupCOVID', 'YHR', 'YHR_low', 'YHR_high', \
        'HFR_low', 'HFR_high']].drop_duplicates().set_index('GroupCOVID')\
        .to_dict()
    Symp_H_Ratio_dict = df_US_dict['YHR']
    Symp_H_Ratio_L_dict = df_US_dict['YHR_low']
    Symp_H_Ratio_H_dict = df_US_dict['YHR_high']
    Hosp_F_Ratio_L_dict = df_US_dict['HFR_low']
    Hosp_F_Ratio_H_dict = df_US_dict['HFR_high']
    
    if run_subgroup is not None:
        if run_subgroup in ['Grocery','Construction']: 
            n_age_orig = n_age
            n_age = 5
            
        elif run_subgroup == 'Teachers':
            n_age_orig = n_age
            n_age = 14
        
    # modify - have to manually calculate if sub = Teachers - how to do?
    if run_subgroup != 'Teachers':
        Symp_H_Ratio = np.array([Symp_H_Ratio_dict[i] for i in \
            age_group_dict[n_age]]) 

        Symp_H_Ratio_w_risk = np.array([[Symp_H_Ratio_L_dict[i] for i in \
            age_group_dict[n_age]], [Symp_H_Ratio_H_dict[i] for i in \
            age_group_dict[n_age]]])

        Hosp_F_Ratio_w_risk = np.array([[Hosp_F_Ratio_L_dict[i] for i in \
            age_group_dict[n_age]], [Hosp_F_Ratio_H_dict[i] for i in \
            age_group_dict[n_age]]])

    if run_subgroup == 'Teachers': # do I need to change something here?
        Symp_H_Ratio = np.array([Symp_H_Ratio_dict[i.split()[0]] for i in \
            age_group_dict[n_age]]) # same calculation, repeat #s for split age groups (14 entries)

        Symp_H_Ratio_w_risk = np.array([[Symp_H_Ratio_L_dict[i.split()[0]] for i in \
            age_group_dict[n_age]], [Symp_H_Ratio_H_dict[i.split()[0]] for i in \
            age_group_dict[n_age]]])

        Hosp_F_Ratio_w_risk = np.array([[Hosp_F_Ratio_L_dict[i.split()[0]] for i in \
            age_group_dict[n_age]], [Hosp_F_Ratio_H_dict[i.split()[0]] for i in \
            age_group_dict[n_age]]])
    
    if run_subgroup is not None:
        if run_subgroup in ['Grocery','Construction']: # not using for Teachers
            # Adjust the ratios, add the 18-49 values
            n_age = n_age_orig
            group_index = age_group_dict[5].index('18-49')
            Symp_H_Ratio = np.append(Symp_H_Ratio,Symp_H_Ratio[group_index])
            Symp_H_Ratio_w_risk = np.append(Symp_H_Ratio_w_risk,\
                Symp_H_Ratio_w_risk[:,group_index].reshape((2,1)),axis=1)
            Hosp_F_Ratio_w_risk = np.append(Hosp_F_Ratio_w_risk,\
                Hosp_F_Ratio_w_risk[:,group_index].reshape((2,1)),axis=1)
        

    # City population
    df = pd.read_csv(data_folder + population_filename_dict[n_age],
        index_col=False)
    pop_metro = np.zeros(shape=(n_age, n_risk))
    for r in range(n_risk):
        #print(df['RiskGroup'] == r)
        #print(age_group_dict[n_age])
        pop_metro[:, r] = df.loc[df['RiskGroup'] == r, age_group_dict[n_age]]\
            .values.reshape(-1)

    # Transmission adjustment multiplier per day and per metropolitan area
    df_school_calendar = pd.read_csv(data_folder + school_calendar_filename, # does the school calendar need to be changed?
        index_col=False)
    school_calendar = df_school_calendar['Calendar'].values.reshape(-1)
    school_calendar_start_date = dt.datetime.strptime(np.\
        str(df_school_calendar['Date'][0]), '%m/%d/%y')

    df_school_calendar_aug = df_school_calendar[df_school_calendar['Date']\
        .str[0].astype(int) >= 8]
    fall_start_date = df_school_calendar_aug[df_school_calendar_aug['Calendar']\
        == 1].Date.to_list()[0]
    fall_start_date = '20200' + fall_start_date.split('/')[0] + \
        fall_start_date.split('/')[1]

    # Contact matrices
    # phi - name for matrix from excel files
    if run_subgroup is not None:
        phi_path = data_folder + run_subgroup + '/Contact matrices with '
        if run_subgroup == 'Grocery':
            phi_path += 'grocery workers.xlsx'
        elif run_subgroup == 'Construction':
            phi_path += 'construction workers.xlsx'

        '''look at format of pop and matrices spreadsheets'''
       
        if run_subgroup != 'Teachers':
            load_phi = lambda sheet: pd.read_excel(phi_path,sheet_name=sheet,
                index=False).values[:,1:]
            phi_school = load_phi('School')
            phi_work = load_phi('Work')
            phi_home = load_phi('Home')
            phi_other = load_phi('Other')
            # check to see where it's used
            phi_all = phi_home + phi_work + phi_other + phi_school

        elif run_subgroup == 'Teachers':

            phi_school = pd.read_csv('Teacher Matrices R/School Contact Matrix 14x14_2.csv', index_col=0).values
            phi_work = pd.read_csv('Teacher Matrices R/Work Contact Matrix 14x14_1.csv', index_col=0).values
            phi_home = pd.read_csv('Teacher Matrices R/Home Contact Matrix 14x14_1.csv', index_col=0).values
            phi_other = pd.read_csv('Teacher Matrices R/Other Contact Matrix 14x14_1.csv', index_col=0).values
            phi_all = phi_home + phi_work + phi_other + phi_school #load_phi('Total') # not used for teachers - w/o subgroups, used to calc other

        phi = {
            'phi_all': phi_all,
            'phi_school': phi_school,
            'phi_work': phi_work,
            'phi_home': phi_home,
            'phi_other': phi_other
        }
        
        # Extra matrix for grocery workers
        if run_subgroup == 'Grocery':
            phi_g_store = load_phi('GroceryStore')
            phi.update({'phi_g_store':phi_g_store})
        
    else:
        phi_all = pd.read_csv(data_folder + \
            contact_matrix_all_filename_dict[n_age], header=None).values
        phi_school = pd.read_csv(data_folder + \
            contact_matrix_school_filename_dict[n_age], header=None).values
        phi_work = pd.read_csv(data_folder + \
            contact_matrix_work_filename_dict[n_age], header=None).values
        phi_home = pd.read_csv(data_folder + \
            contact_matrix_home_filename_dict[n_age], header=None).values

        phi = {
            'phi_all': phi_all,
            'phi_school': phi_school,
            'phi_work': phi_work,
            'phi_home': phi_home
        }

    return age_group_dict, pop_metro, school_calendar, \
        school_calendar_start_date, fall_start_date, phi, Symp_H_Ratio, \
        Symp_H_Ratio_w_risk, Hosp_F_Ratio_w_risk

def SEIR_get_param(symp_h_ratio_overall, symp_h_ratio, hosp_f_ratio, n_age,
                   n_risk, deterministic=True):
    """ Get epidemiological parameters
    :param symp_h_ratio_overall: np.array of shape (n_age, )
    :param symp_h_ratio: np.array of shape (n_risk, n_age)
    :param hosp_f_ratio: np.array of shape (n_age, )
    :param n_age: int, number of age groups
    :param n_risk: int, number of risk groups
    :param deterministic: boolean, whether to remove parameter stochasticity
    """
    # which index in pop array is 65+
    # change to reflect multiple 65+ age groups - make a list of indices if N_AGE = 14 or sub = Teachers
    index_65 = -1 if n_age == 5 else -2

    
    time_h_to_r = np.ones(n_age) * T_H_TO_R
    time_h_to_d = np.ones(n_age) * T_H_TO_D

    # get indices and modify the 65+ groups here
    time_h_to_r[index_65] = T_H_TO_R_65
    time_h_to_d[index_65] = T_H_TO_D_65

    r0 = R0
    double_time = DOUBLE_TIME

    gamma_h = 1 / T_H_TO_R
    gamma_y_c = 1 / np.median(T_Y_TO_R_PARA)
    if deterministic:
        gamma_y = gamma_y_c
    else:
        gamma_y = 1 / T_Y_TO_R_DIST(T_Y_TO_R_PARA)
    gamma_a = gamma_y
    gamma = np.array([gamma_a * np.ones(n_age), gamma_y * np.ones(n_age),
        gamma_h * np.ones(n_age)])

    # Incubation period, non-infectious (before pre-symptomatic)
    sigma_c = 1 / np.median(T_EXPOSED_PARA) * np.ones(n_age)
    if deterministic:
        sigma = sigma_c
    else:
        sigma = 1 / T_EXPOSED_DIST(T_EXPOSED_PARA) * np.ones(n_age)
        
    rho_a = 1 / T_PA_TO_IA * np.ones(n_age)
    rho_y = 1 / T_PY_TO_IY * np.ones(n_age)
    rho = np.array([rho_a, rho_y])

    # Rate to get to hospital
    eta = 1 / T_ONSET_TO_H * np.ones(n_age)

    # Hospital rate for non-survivors
    mu = 1 / T_H_TO_D * np.ones(n_age)

    nu = hosp_f_ratio * gamma_h / (mu + (gamma_h - mu) * hosp_f_ratio)

    pi = symp_h_ratio * gamma_y / (eta + (gamma_y - eta) * symp_h_ratio)

    tau = (1 - ASYMP_RATE) * np.ones(n_age)
    
    
    # Relative infectiousness
    omega_a = ASYMP_RELATIVE_INFECT
    omega_y = 1. # by definition
    omega_h = 0. # by definition
    
    # Relative infectiousness of pre-symptomatic compartments
    omega_p = PROP_TRANS_IN_P / (1-PROP_TRANS_IN_P) *\
        (tau * omega_y * (symp_h_ratio_overall / eta + (1 - \
        symp_h_ratio_overall) / gamma_y) +\
         (1-tau) * omega_a / gamma_a) / \
        ((tau * omega_y / rho_y + (1-tau) * omega_a / rho_a))
    
    omega_py = omega_p * omega_y
    omega_pa = omega_p * omega_a
    
    omega = np.array([omega_a * np.ones(n_age),
                      omega_y * np.ones(n_age),
                      omega_h * np.ones(n_age),
                      omega_pa,
                      omega_py])

    para = {
        'r0': r0,
        'double_time': double_time,
        'gamma': gamma,
        'sigma': sigma,
        'eta': eta,
        'mu': mu,
        'nu': nu,
        'pi': pi,
        'tau': tau,
        'omega': omega,
        'rho': rho
    }

    return para
