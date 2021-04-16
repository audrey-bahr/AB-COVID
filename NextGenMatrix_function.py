# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:26:13 2020

@author: remyp
"""
import os
import numpy as np
import pandas as pd
from numpy import linalg as LA

###############################################################################
def compute_contact_mat(home_mat,school_mat,work_mat,other_mat,
                        schools='closed',c_reduc=0.0,
                        c_reduc_schools=0.0,time='avg',
                        CW=False,delta_cw=1.):
    """ Compute total contact matrix based on matrices at different locations
        schools: closed or open. Whether schools are closed (removed from total
    contacts) or open
        c_reduc: number between 0 and 1. Reduction in all contacts
    (home, work, others and schools if open)
        c_reduc_schools: number between 0 and 1. Reduction in school contacts
    on top of other contact reduction
        time: week, weekend or avg. Whether matrix during a weekday, weekend
    or the average
        CW: whether contact workers are the last group, to be modeled differently
        delta_cw: multiplier for construction workers work contacts
    """
    if schools == 'open':
        schools_m = 1
    elif schools == 'closed':
        schools_m = 0
    else:
        print('Wrong value for schools')
        schools_m = -1000
    
    
    week_C = (1-c_reduc) * (home_mat + \
                            (1-c_reduc_schools) * schools_m * school_mat + \
                            work_mat + \
                            other_mat)
    weekend_C = (1-c_reduc) * (home_mat + other_mat)
    avg_C = week_C*5/7 + weekend_C*2/7
    
    # Adjustment for construction workers
    if CW:
        wi = len(home_mat)-1 # last group
        CW_week = (1-c_reduc) * (home_mat[wi,wi] + schools_m*school_mat[wi,wi] +\
            other_mat[wi,wi]) + delta_cw * work_mat[wi,wi]
        CW_weekend = (1-c_reduc) * (home_mat[wi,wi] + other_mat[wi,wi])
        avg_CW = CW_week*5/7 + CW_weekend*2/7
        week_C[wi,wi] = CW_week
        weekend_C[wi,wi] = CW_weekend
        avg_C[wi,wi] = avg_CW
    
    if time == 'week':
        outC = week_C
    elif time == 'weekend':
        outC = weekend_C
    elif time == 'avg':
        outC = avg_C
    else:
        print('Wrong value for time')
        outC = -1000
    
    return outC
###############################################################################
def NGM_R0(p_Hosp,T_ONSET_TO_H,dE,dP,dI,dA,tau,omega_y,omega_a,preS,
           avg_contact_mat,pop_array,beta,S_prop=None):
    
    # Number of age groups
    n_age = avg_contact_mat.shape[0] # n_age = 5
    
    ## Derived variables
    # Effective symptomatic infectious period
    dY = np.array([x*T_ONSET_TO_H + (1-x)*dI for x in p_Hosp])
    
    # Pre-symptomatic relative infectiousness
    omega_p = preS / (1-preS) * (tau * omega_y * dY + (1-tau) * omega_a * dA) / \
        (dP * (tau * omega_y + (1-tau) * omega_a))
    
    # Relative infectiousness of pre-symptomatic compartments
    omega_py = omega_p * omega_y
    omega_pa = omega_p * omega_a
    
    
    # Rates between compartments
    sigma = 1/dE * np.ones(n_age)
    rho_a = 1 / dP * np.ones(n_age)
    rho_y = 1 / dP * np.ones(n_age)
    gamma_a = 1 / dA * np.ones(n_age)
    gamma_y = 1/dY # effective total rate
    
    
    # Population proportion ratios n_i / n_j multiplies each cell T_ij
    n_pop = pop_array/sum(pop_array)
    n_pop_ratio = np.dot(np.reshape(n_pop,(n_age,1)),np.reshape(1/n_pop,(1,n_age)))
    
    # Proportion of population susceptible in each group
    if S_prop is None:
        S_prop = np.ones(n_age)
    S_prop_m = np.tile(S_prop,(n_age,1)).T
    

    ########
    ## Matrix construction
    # Matrices are block matrices
    # First 5 rows columns correspond to E, next 5 to A, last 5 to Y
    n_compartments = 5 # nb compartments in NGM, x=[E,P^A,P^Y,I^A,I^Y]
    big_C = np.tile(avg_contact_mat,(n_compartments,n_compartments))
    Zm = np.zeros((n_age,n_age)) # zero matrix, 
    
    # Transmission and transition matrices
    T_mat = big_C*np.block([[
        np.zeros((n_age,n_age)) * S_prop_m,
        beta * n_pop_ratio * np.tile(omega_pa,(n_age,1)) * S_prop_m,
        beta * n_pop_ratio * np.tile(omega_py,(n_age,1)) * S_prop_m,
        beta * n_pop_ratio * np.tile(omega_a,(n_age,n_age)) * S_prop_m,
        beta * n_pop_ratio * np.tile(omega_y,(n_age,n_age)) * S_prop_m],
        [np.zeros((4*n_age,5*n_age))]])
    
    sigma_mat = np.block([
        [np.diag(-sigma), np.zeros((n_age,4*n_age))],
        [np.diag((1-tau)*sigma), np.diag(-rho_a), Zm, Zm, Zm],
        [np.diag(tau*sigma), Zm, np.diag(-rho_y), Zm, Zm],
        [Zm, np.diag(rho_a), Zm, np.diag(-gamma_a),Zm],
        [Zm, Zm, np.diag(rho_y), Zm, np.diag(-gamma_y)]])
    
    
    
    #######
    ## R0 computation from NGM
    K_L = np.dot(-T_mat,np.linalg.inv(sigma_mat))
    w_KL, v_KL = LA.eig(K_L)
    R0_KL = np.max(w_KL) # 4.126799171906285 with current inputs
    # print('R0 estimated from NGM',R0_KL)
        
    ########
    ## Rough R0, approximation: 3.87121575122189 with current inputs
    c_avg_all = np.dot(avg_contact_mat.sum(axis=1),n_pop)
    rough_r0 = beta * c_avg_all * ((1-tau)*np.mean(omega_pa)/np.mean(rho_a) +\
                                   tau*np.mean(omega_py)/np.mean(rho_y)  +\
                                   (1-tau)*omega_a/np.mean(gamma_a) +\
                                   tau*omega_y/np.mean(gamma_y))
    
    ##### Keeling & Rohani page 60: Box 3.1
    ## Compute the growth rate
    J_mat = T_mat+sigma_mat
    w_J, v_J = LA.eig(J_mat)
    lambda_J = np.max(w_J)
    
    return R0_KL, lambda_J
###############################################################################
###### Specific contact matrices
# Folders and files
FOLDER_L = ['C:','Users','remyp','Research','COVID-19','TeacherRisk','NGM']
DIR_FOLDER = os.sep.join(FOLDER_L)
os.chdir(DIR_FOLDER)
input_folder = 'Matrices'
input_path = os.sep.join([DIR_FOLDER,input_folder,''])

## Get contact matrices
school_org = pd.read_csv(input_path + 'School Contact Matrix 14x14_2.csv', index_col=0).values
work_org = pd.read_csv(input_path + 'Work Contact Matrix 14x14_1.csv', index_col=0).values
home_org = pd.read_csv(input_path + 'Home Contact Matrix 14x14_1.csv', index_col=0).values
other_org = pd.read_csv(input_path + 'Other Contact Matrix 14x14_1.csv', index_col=0).values
phi_all = home_org + work_org + other_org + school_org #load_phi('Total') # not used for teachers - w/o subgroups, used to calc other

# Population in each age group
pop_path = input_path + 'Austin Population - 6 age groups, school employment.csv'
pop_org = pd.read_csv(pop_path).iloc[:,2:].values
pop_array = pop_org.sum(axis=0)

###### Inputs
# Probability of hospitalization
p_Hosp = np.array([0.0004    , 0.0004    ,\
                   0.00848034, 0.00848034, 0.00848034,\
                   0.03349423, 0.03349423, 0.03349423,\
                   0.09308001, 0.09308001, 0.09308001,\
                   0.14558845, 0.14558845, 0.14558845])

T_ONSET_TO_H = 5.9 # time from symptom onset to hospitalization
dInc = 5.2 # incubation period
dP = 2.3 # pre-symptomatic infectious period
dE = dInc - dP # non infectious part of the incubation period
dI = 4. # infectious period (no hospitalization)
dA = dI # asymptomatic infectious period


# Symptomatic proportion
tau = 0.57

# Contact reduction from order
sd_order = 0.0 # overall contact reduction (all locations, including schools)
sd_school = 0.0 # contact reduction in schools only

# Average contact matrix: schools open, no contact reduction, average
avg_contact_mat = compute_contact_mat(home_org,school_org,work_org,other_org,
                        schools='open',c_reduc=0.0,
                        c_reduc_schools=sd_school,time='avg')

# Average contact matrix: schools closed and stay home order
reduced_contact_mat = compute_contact_mat(home_org,school_org,work_org,other_org,
                        schools='closed',c_reduc=sd_order,
                        c_reduc_schools=sd_school,time='avg')


# Relative infectiousness
omega_y = 1.
omega_a = 2/3

# Proportion of infections pre-symptomatic
preS = 0.44

# Beta from fitting
# beta = 0.05 # R0 of 3.9
beta = 0.01280706 # R0 of 1.00

# Proportion of population susceptible in each age group
# S_prop = [1.0, 0.9, 0.8, 1.0, 0.0]
S_prop = [1.] * len(pop_array)

### Compute R0
R0, lambd = NGM_R0(p_Hosp,T_ONSET_TO_H,dE,dP,dI,dA,tau,omega_y,omega_a,preS,
           avg_contact_mat,pop_array,beta,S_prop=S_prop)
print('R0:',R0)
print('Growth Rate:',lambd)

### Compute R0 with contact reduction
R0_reduc, lambd_reduc = NGM_R0(p_Hosp,T_ONSET_TO_H,dE,dP,dI,dA,tau,omega_y,omega_a,preS,
           reduced_contact_mat,pop_array,beta,S_prop=S_prop)
print('R0:',R0_reduc)
print('Growth Rate:',lambd_reduc)


###############################################################################




