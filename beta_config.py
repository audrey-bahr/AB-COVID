import numpy as np
import SEIR_param_publish

# Choose whether to run simulations for a subgroup
run_subgroup = 'Teachers'

# Parameters specific to subgroups
if run_subgroup == 'Grocery':
    N_AGE = 6
    
    # Social distancing at the grocery store to reduce effective contacts there
    # for instance masks of 6 feet distancing
    g_shopper_sd_list = [0,0.5,0.75,0.9] # 0 means no reduction, 1 no contacts
    g_worker_sd_list = [0,0.5,0.75,0.9] # 0 means no reduction, 1 no contacts

    subgroup_param = {
        'g_shopper_sd':g_shopper_sd_list,
        'g_worker_sd':g_worker_sd_list
    }
    
elif run_subgroup == 'Construction':
    N_AGE = 6
    
    # Increase/Decrease in construction workers work contacts among themselves
    delta_CW_list = [0.5,1,2]
    
    # Proportion of all construction workers allowed to work
    prop_CW_list = [0, 0.25, 0.50, 0.75, 1]
    
    subgroup_param = {
        'delta_CW':delta_CW_list,
        'prop_CW':prop_CW_list
    }

# added
elif run_subgroup == 'Teachers':
    # adult age groups split into employment categories result in > 6 groups
    # size of matrices is 14x14
    N_AGE = 14 # with t, s, v and not x

    # proportion of people in school (s, t, x)
    prop_school_sd = [0, 0.25, 0.50, 0.75, 1]
    #prop_gen_sd # 1 value, (line 95)
    # susceptibility/infectiousness parameters
    suscep_list = [[1.25, 1.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    # should still result in 5 sims - list of length 1; change first two values for kids (1-1.25) and 1s for the rest
    infect_list = [[1.25, 1.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
  
    # add to dictionary

    subgroup_param = {
        'prop_school_sd':prop_school_sd,
        'suscep_param':suscep_list,
        'infect_param':infect_list
    }
    
    
else:
    N_AGE = 5  # number of age groups

N_RISK = 2  # high, low
CITY = 'AustinMSA'

# Get input data in Excel files
DATA_FOLDER = "Austin Data/"

GROWTH_RATE_LIST = ['high']

''' any of these need to be changed? FallStartDate? - change later '''
AgeGroupDict, MetroPop, SchoolCalendar, TimeBegin, FallStartDate, Phi, \
    SympHratioOverall, SympHratio, HospFratio = SEIR_param_publish.\
    SEIR_get_data(DATA_FOLDER, CITY, N_AGE, N_RISK, run_subgroup)


# Simulation parameters
SHIFT_WEEK = 0
TIME_BEGIN_SIM = 20200215 # 20210105 # AISD start date
# Time intervals in each day, determines time-step
INTERVAL_PER_DAY = 10
NUM_SIM_FIT = 1
NUM_SIM = 1
TOTAL_TIME = 7 * 25
VERBOSE = False

# School closure parameters
TRIGGER_TYPE = 'cml'
CLOSE_TRIGGER_LIST = ['date__20300319']    # '''change - 20220319? school closure date'''
REOPEN_TRIGGER_LIST = ['no_na_' + FallStartDate]
SD_DATE_FIT = [20200324, 20200818]
SD_LEVEL_LIST_FIT = [0.5] # fitted up to 4/8
SD_DATE = [20300324, 20400101]
SD_LEVEL_LIST = [0.746873309820472] # fitted up to 4/19 # general population parameter
MONITOR_LAG = 0  # 0 days monitor lag
REPORT_RATE = 1.

# Epidemiological parameters
BETA0_dict = {'high': 0.0230246324976167} # fitted up to 4/19 # make lower - 0.0530246324976167 to 

    #'''start_condition - corresponds to infected in each age group
    #    start with >1 infected
    #    may depend on start date - actual data for Jan 2021''''
START_CONDITION = 1

# initial infections
# start condition - 1 infected in each group?
# what are the rows/columns here?
I0 = np.array([[0, 0], [0, 0], [START_CONDITION, 0], [0, 0], [0, 0]])

if run_subgroup is not None:
    if run_subgroup in ['Grocery','Construction']: # added Teacher
        I0 = np.append(I0,np.zeros((1,N_RISK)),axis=0)
    # format for I0 - 14 pairs 
    elif run_subgroup == 'Teachers':
        I0 = np.append(I0,np.zeros((9,N_RISK)),axis=0) 
        
