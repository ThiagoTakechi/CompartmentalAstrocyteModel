# ----------------------------------------------------------------------------
# Contributors: Thiago O. Bezerra
#               Antonio C. Roque
# ----------------------------------------------------------------------------
# File description:
#
# Functions implementing the current densities and differential equations for 
# each variable in the astrocyte model, to calculate the 
# pre-synaptic spike times, and implement the 4th-order Runge-Kutta method to
# solve the system of differential equations.
# ----------------------------------------------------------------------------

from numba import njit 
from numpy import exp, sqrt, vstack, ceil, ones, zeros
from numpy.random import exponential

@njit
def diffusion(M, D_coeff, mol):        
    return D_coeff*(M.dot(mol) - M.sum(axis=1)*mol)

@njit
def f_prod_PLCb_glu(v_beta, Glu_o, alpha, K_R, K_p, Ca_i, K_pi):
    return v_beta * Glu_o ** alpha / (Glu_o ** alpha + (K_R + K_p * Ca_i / (Ca_i + K_pi)) ** alpha)

@njit
def f_prod_PLCb_DA(v_DA, DA_o, beta, K_DA, K_p, Ca_i, K_pi):
     return v_DA * DA_o ** beta / (DA_o ** beta + (K_DA + K_p * Ca_i / (Ca_i + K_pi))**beta)

@njit
def f_prod_PLCd(v_delta, IP3, kappa_delta, Ca_i, K_PLCdelta): 
    return v_delta / (1 + IP3/kappa_delta) * Ca_i ** 2 / (Ca_i ** 2 + K_PLCdelta ** 2)

@njit
def f_degr_IP3_3K(v_3K, Ca_i, K_D, IP3, K_3):
    return v_3K * Ca_i ** 4 / (Ca_i ** 4 + K_D ** 4) * IP3 / (IP3 + K_3)

@njit
def f_degr_IP_5P(r_5P, IP3):
    return r_5P * IP3

@njit
def f_J_IP3R (F, A, Vol, r_C, IP3, d_1, Ca_i, d_5, h, Ca_ER):
    return F * Vol / A * r_C * (IP3 / (IP3 + d_1)) ** 3 * (Ca_i / (Ca_i + d_5)) ** 3 * h ** 3 * (Ca_ER - Ca_i)

@njit
def f_J_CERleak(F, A, Vol, r_L, Ca_ER, Ca_i):
    return F * Vol / A * r_L * (Ca_ER - Ca_i)

@njit
def f_J_SERCA(F, A, Vol, v_ER, Ca_i, K_ER):
    return F * Vol / A * v_ER * Ca_i ** 2 / (Ca_i ** 2 + K_ER ** 2)

@njit
def f_J_GluT(J_GluTmax, K_i, K_GluTmK, Na_o, K_GluTmN, Glu_o, K_GluTmg):
    return J_GluTmax * (K_i / (K_i + K_GluTmK)) * (Na_o**3 / (Na_o**3 + K_GluTmN**3)) * (Glu_o / (Glu_o + K_GluTmg))

@njit
def f_J_NKA(J_NKAmax, Na_i, K_NKAmN, K_o, K_NKAmK):
    return J_NKAmax * (Na_i**1.5 / (Na_i**1.5 + K_NKAmN ** 1.5)) * (K_o / (K_o + K_NKAmK))

@njit
def f_J_NCX(J_NCXmax, Na_o, K_NCXmN, Ca_o, K_NCXmC, Na_i, eta, V, F, R, T, Ca_i, k_sat):

    alpha = J_NCXmax * (Na_o**3 / (Na_o**3 + K_NCXmN**3)) * (Ca_o / (Ca_o + K_NCXmC))
    beta = Na_i**3 / Na_o**3 * exp(eta * V * F / (R * T)) - Ca_i / Ca_o * exp((eta - 1)* V * F / (R * T))
    gama = 1 + k_sat * exp((eta - 1)* V * F / (R * T))

    return  alpha * beta / gama

@njit
def f_J_Naleak(g_Naleak, V, E_Na):
    return g_Naleak * (V - E_Na)

@njit
def f_J_Kleak(g_Kleak, V, E_K):
    return g_Kleak * (V - E_K)

### ODEs ###

@njit
def dCa_odt(M, p, A, Vol, Ca_o, Na_o, Ca_i, Na_i, V):

    J_diff_Cao  = diffusion(M, p['D_Cao'], Ca_o)
    
    J_NCX     = f_J_NCX(p['J_NCXmax'], Na_o, p['K_NCXmN'], Ca_o, p['K_NCXmC'], Na_i, p['eta'], V, p['F'], p['R'], p['T'], Ca_i, p['k_sat'])
    
    return - A / (Vol * p['F']) * J_NCX + J_diff_Cao

@njit
def dNa_odt(M, p, A, Vol, Ca_o, Na_o, K_o, Ca_i, Na_i, K_i, V, Glu_o):
    
    
    J_diff_Nao  = diffusion(M, p['D_Nao'], Na_o)
    
    J_GluT   = f_J_GluT(p['J_GluTmax'], K_i, p['K_GluTmK'], Na_o, p['K_GluTmN'], Glu_o, p['K_GluTmg'])
    J_NKA    = f_J_NKA(p['J_NKAmax'], Na_i, p['K_NKAmN'], K_o, p['K_NKAmK'])
    J_NCX    = f_J_NCX(p['J_NCXmax'], Na_o, p['K_NCXmN'], Ca_o, p['K_NCXmC'], Na_i, p['eta'], V, p['F'], p['R'], p['T'], Ca_i, p['k_sat'])
    J_Naleak = f_J_Naleak(p['g_Naleak'], V, p['E_Na'])
    
    return - A / (Vol * p['F']) * (3*J_GluT - 3*J_NKA - 3*J_NCX - J_Naleak) + J_diff_Nao

@njit
def dK_odt(M, p, A, Vol, Na_o, K_o, Na_i, K_i, V, Glu_o): 
    
    
    J_diff_Ko = diffusion(M, p['D_Ko'], K_o)
    
    J_GluT  = f_J_GluT(p['J_GluTmax'], K_i, p['K_GluTmK'], Na_o, p['K_GluTmN'], Glu_o, p['K_GluTmg'])
    J_NKA   = f_J_NKA(p['J_NKAmax'], Na_i, p['K_NKAmN'], K_o, p['K_NKAmK'])
    J_Kleak = f_J_Kleak(p['g_Kleak'], V, p['E_K'])
        
    return - A / (Vol * p['F']) * (-J_GluT + 2*J_NKA - J_Kleak) + J_diff_Ko

@njit
def dCa_idt(M, p, A, Vol, ratio_ER, Ca_o, Na_o, Ca_i, h, IP3, Ca_ER, Na_i, V): 
    
    J_diff_Cai  = diffusion(M, p['D_Ca'], Ca_i)
    
    J_NCX     = f_J_NCX(p['J_NCXmax'], Na_o, p['K_NCXmN'], Ca_o, p['K_NCXmC'], Na_i, p['eta'], V, p['F'], p['R'], p['T'], Ca_i, p['k_sat'])
    J_IP3R    = f_J_IP3R (p['F'], A, Vol, p['r_C'], IP3, p['d_1'], Ca_i, p['d_5'], h, Ca_ER)
    J_SERCA   = f_J_SERCA(p['F'], A, Vol, p['v_ER'], Ca_i, p['K_ER'])
    J_CERleak = f_J_CERleak(p['F'], A, Vol, p['r_L'], Ca_ER, Ca_i)
    
    return A / (Vol * p['F']) * J_NCX + A * sqrt(ratio_ER) / (Vol * p['F']) * (J_IP3R - J_SERCA + J_CERleak) + J_diff_Cai

@njit
def dhdt(p, Ca_i, h, IP3):
    return p['a_2']*(p['d_2'] * (IP3 + p['d_1']) / (IP3 + p['d_3']) * (1 - h) - Ca_i * h)

@njit
def dIP3dt(M, p, Ca_i, IP3, Glu_o, DA_o):
    
    
    J_diff_IP3  = diffusion(M, p['D_IP3'], IP3)
    
    prod_PLCb_glu = f_prod_PLCb_glu(p['v_beta'], Glu_o, p['alpha'], p['K_R'], p['K_p'], Ca_i, p['K_pi'])
    prod_PLCd     = f_prod_PLCb_DA(p['v_DA'], DA_o, p['beta'], p['K_DA'], p['K_p'], Ca_i, p['K_pi'])
    prod_PLCb_DA  = f_prod_PLCd(p['v_delta'], IP3, p['kappa_delta'], Ca_i, p['K_PLCdelta'])
    degr_IP3_3K   = f_degr_IP3_3K(p['v_3K'], Ca_i, p['K_D'], IP3, p['K_3'])
    degr_IP_5P    = f_degr_IP_5P(p['r_5P'], IP3)
    
    return prod_PLCb_glu + prod_PLCd + prod_PLCb_DA - degr_IP3_3K - degr_IP_5P + J_diff_IP3

@njit
def dCa_ERdt(M, p, A, Vol, ratio_ER, Ca_i, h, IP3, Ca_ER):
    
    J_diff_CaER = diffusion(M, p['D_CaER'], Ca_ER)
    
    J_IP3R    = f_J_IP3R (p['F'], A, Vol, p['r_C'], IP3, p['d_1'], Ca_i, p['d_5'], h, Ca_ER)
    J_SERCA   = f_J_SERCA(p['F'], A, Vol, p['v_ER'], Ca_i, p['K_ER'])
    J_CERleak = f_J_CERleak(p['F'], A, Vol, p['r_L'], Ca_ER, Ca_i)
    
    return A * sqrt(ratio_ER) / (Vol * p['F'] * ratio_ER) * (-J_IP3R + J_SERCA - J_CERleak) + J_diff_CaER

@njit
def dNa_idt(M, p, A, Vol, Ca_o, Na_o, K_o, Ca_i, Na_i, K_i, V, Glu_o):
    
    J_diff_Nai = diffusion(M, p['D_Na'], Na_i)
    
    J_GluT   = f_J_GluT(p['J_GluTmax'], K_i, p['K_GluTmK'], Na_o, p['K_GluTmN'], Glu_o, p['K_GluTmg'])
    J_NKA    = f_J_NKA(p['J_NKAmax'], Na_i, p['K_NKAmN'], K_o, p['K_NKAmK'])
    J_NCX    = f_J_NCX(p['J_NCXmax'], Na_o, p['K_NCXmN'], Ca_o, p['K_NCXmC'], Na_i, p['eta'], V, p['F'], p['R'], p['T'], Ca_i, p['k_sat'])
    J_Naleak = f_J_Naleak(p['g_Naleak'], V, p['E_Na'])
    
    return A / (Vol * p['F']) * (3*J_GluT - 3*J_NKA - 3*J_NCX - J_Naleak) + J_diff_Nai

@njit
def dK_idt(M, p, A, Vol, Na_o, K_o, Na_i, K_i, V, Glu_o):
    
    J_diff_Ki = diffusion(M, p['D_K'], K_i)
    
    J_GluT  = f_J_GluT(p['J_GluTmax'], K_i, p['K_GluTmK'], Na_o, p['K_GluTmN'], Glu_o, p['K_GluTmg'])
    J_NKA   = f_J_NKA(p['J_NKAmax'], Na_i, p['K_NKAmN'], K_o, p['K_NKAmK'])
    J_Kleak = f_J_Kleak(p['g_Kleak'], V, p['E_K'])
        
    return A / (Vol * p['F']) * (-J_GluT + 2*J_NKA - J_Kleak) + J_diff_Ki

@njit
def dVdt(p, A, Vol, Ca_o, Na_o, K_o, Ca_i, h, IP3, Ca_ER, Na_i, K_i, V, Glu_o):
    
    J_IP3R    = f_J_IP3R (p['F'], A, Vol, p['r_C'], IP3, p['d_1'], Ca_i, p['d_5'], h, Ca_ER)
    J_SERCA   = f_J_SERCA(p['F'], A, Vol, p['v_ER'], Ca_i, p['K_ER'])
    J_CERleak = f_J_CERleak(p['F'], A, Vol, p['r_L'], Ca_ER, Ca_i)
    J_NCX = f_J_NCX(p['J_NCXmax'], Na_o, p['K_NCXmN'], Ca_o, p['K_NCXmC'], Na_i, p['eta'], V, p['F'], p['R'], p['T'], Ca_i, p['k_sat'])
    J_GluT = f_J_GluT(p['J_GluTmax'], K_i, p['K_GluTmK'], Na_o, p['K_GluTmN'], Glu_o, p['K_GluTmg'])
    J_NKA = f_J_NKA(p['J_NKAmax'], Na_i, p['K_NKAmN'], K_o, p['K_NKAmK'])
    J_Naleak = f_J_Naleak(p['g_Naleak'], V, p['E_Na'])
    J_Kleak = f_J_Kleak(p['g_Kleak'], V, p['E_K'])
    
    return -1 / p['C_m'] * (-2*J_IP3R + 2*J_SERCA - 2*J_CERleak + J_NCX - 2*J_GluT + J_NKA + J_Naleak + J_Kleak)

@njit
def dGlu_odt(M, p, Glu_o):
    
    J_diff_Gluo = diffusion(M, p['D_glu'], Glu_o)
    
    return -p['G_glu'] * Glu_o + J_diff_Gluo

@njit
def dDA_odt(M, p, DA_o):
    
    J_diff_DAo = diffusion(M, p['D_DA'], DA_o)
    
    return - p['G_DA'] * DA_o + J_diff_DAo

@njit
def model_eqs(M, p, A, Vol, ratio_ER, Ca_o, Na_o, K_o, Ca_i, h, IP3, Ca_ER, Na_i, K_i, V, Glu_o, DA_o):
    return vstack((dCa_odt(M, p, A, Vol, Ca_o, Na_o, Ca_i, Na_i, V),
                     dNa_odt(M, p, A, Vol, Ca_o, Na_o, K_o, Ca_i, Na_i, K_i, V, Glu_o),
                     dK_odt(M, p, A, Vol, Na_o, K_o, Na_i, K_i, V, Glu_o),
                     dCa_idt(M, p, A, Vol, ratio_ER, Ca_o, Na_o, Ca_i, h, IP3, Ca_ER, Na_i, V),
                     dhdt(p, Ca_i, h, IP3),
                     dIP3dt(M, p, Ca_i, IP3, Glu_o, DA_o),
                     dCa_ERdt(M, p, A, Vol, ratio_ER, Ca_i, h, IP3, Ca_ER),
                     dNa_idt(M, p, A, Vol, Ca_o, Na_o, K_o, Ca_i, Na_i, K_i, V, Glu_o),
                     dK_idt(M, p, A, Vol, Na_o, K_o, Na_i, K_i, V, Glu_o),
                     dVdt(p, A, Vol, Ca_o, Na_o, K_o, Ca_i, h, IP3, Ca_ER, Na_i, K_i, V, Glu_o),
                     dGlu_odt(M, p, Glu_o),
                     dDA_odt(M, p, DA_o)))

# Stationary Values for initializing variables

def null_IP3(IP3, p):

    alpha = p['v_delta']*p['kappa_delta'] * (p['Ca_irest']**2 / (p['Ca_irest']**2 + p['K_PLCdelta']**2))
    beta = p['v_3K'] * (p['Ca_irest']**4 / (p['Ca_irest']**4 + p['K_D']**4))

    return alpha/(p['kappa_delta'] + IP3) - beta*(IP3 / (p['K_3'] + IP3)) - p['r_5P']*IP3

def null_h(p):

    alpha = p['d_2']*(p['IP3_rest'] + p['d_1']) / (p['IP3_rest'] + p['d_3'])

    return alpha / (alpha + p['Ca_irest'])

def null_ca_er(p):
    
    alpha = p['r_C'] * (p['IP3_rest'] / (p['IP3_rest'] + p['d_1']))**3 * (p['Ca_irest'] / (p['Ca_irest'] + p['d_5']))**3 * p['h_rest']**3
    beta = p['v_ER'] * (p['Ca_irest']**2 / (p['Ca_irest']**2 + p['K_ER']**2))

    return p['Ca_irest'] + beta / (alpha + p['r_L'])

def null_na_i(p):

    J_NKA = f_J_NKA(p['J_NKAmax'], p['Na_irest'], p['K_NKAmN'], p['K_orest'], p['K_NKAmK'])
    J_NCX = f_J_NCX(p['J_NCXmax'], p['Na_orest'], p['K_NCXmN'], p['Ca_orest'], p['K_NCXmC'], p['Na_irest'], p['eta'], p['v_rest'], p['F'], 
                    p['R'], p['T'], p['Ca_irest'], p['k_sat'])

    return -3*(J_NKA + J_NCX) / (p['v_rest'] - p['E_Na'])

def null_k_i(p):

    J_NKA   = f_J_NKA(p['J_NKAmax'], p['Na_irest'], p['K_NKAmN'], p['K_orest'], p['K_NKAmK'])

    return 2*J_NKA / (p['v_rest'] - p['E_K'])

@njit
def calculate_stimuli(neut = (0,), stimuli_times = (0,), t = 0, dt = 1, stimuli_types = (0,), stimuli_t_init = (0,), stimuli_t_end = (0,), 
                      stimuli_Hz = (0,), stimuli_comparts = (0,), rho = (0,)):
    """Simulate the release of neurotransmitter from a pre-synaptic neuron. 
    
    The mode of neurotransmitter release is defined by stimulus type. If the stimulus
    type is poissonian (stimuli_types = "poisson"), the pre-synaptic spike times 
    (neurotransmitter release) are drawn from a poisson (exponential time) and the 
    neurotransmitter concentration in each compartment is incremented by rho. The 
    frequency of the poisson distribution is given by 1/Hz (1/stimuli_Hz). If the 
    stimulus type is constant, the concentration is set to rho.

    Stimuli are applied from stimuli_t_init to stimuli_t_end

    Parameters
    ----------
    neut: list or numpy 1D-array 
        neurotransmition concentration and length equal the number of compartments 
        under stimulation.
    stimuli_times: list or numpy 1D-array
        previous pre-synaptic spike times for each compartment under stimulation.
    t: float
        current time.
    dt: float
        time step in the integration method.
    stimuli_types: list or tuple
        type of stimuli - "poisson", "constant" or none. If neither option were given,
        the neurotransmitter value will be set to zero.
    stimuli_t_init: list, tuple or numpy 1D-array
        initial stimulation time for each compartment under stimulation.
    stimuli_t_end: list, tuple or numpy 1D-array 
        ending stimulation time for each compartment under stimulation.
    stimuli_Hz: list, tuple or numpy 1D-array 
        frequency (in seconds) for each compartment under stimulation.
    stimuli_comparts: list, tuple or numpy 1D-array 
        compartments under stimulation.
    rho: float
        if stimulus type is "poisson", it is the increment that the neurotransmitter 
        receive for each release event. If stimulus_type is "constant", it is the value
        at which the neurotransmitter concentrations is fixed.

    Return
    ----------
    neut: list or numpy 1D-array
        updated neurotransmitter concentrations. Its length equals the number of 
        compartments under stimulation.
    stimuli_times: list or numpy 1D-array
        updated pre-synaptic spike times for poissonian stimulation.
    """

    n_stim = len(stimuli_types)

    for i_stim in range(n_stim):

        stimulus_type  = stimuli_types[i_stim]
        stimulus_compartments = stimuli_comparts[i_stim]
        n_stim_comparts = len(stimulus_compartments)

        if stimulus_type == 'poisson':

            Hz = stimuli_Hz[i_stim]
            
            for i_compart in range(n_stim_comparts):

                compart = int(stimulus_compartments[i_compart])
                t_init = stimuli_t_init[i_stim]
                t_end = stimuli_t_end[i_stim]
                
                if t == 0:
                    stimuli_times[i_stim, compart-1] += ceil(exponential(scale=1/Hz,size=1)/dt)[0] + int(t_init/dt)

                elif (t == stimuli_times[i_stim, compart-1]) & (t >= t_init/dt) & (t <= t_end/dt):
                    neut[compart-1] = neut[compart-1] + rho
                    stimuli_times[i_stim, compart-1] += ceil(exponential(scale=1/Hz,size=1)/dt)[0]

        elif stimulus_type == 'constant':
                
            for i_compart in range(n_stim_comparts):
            
                compart = int(stimulus_compartments[i_compart])
                t_init = stimuli_t_init[i_stim]
                t_end = stimuli_t_end[i_stim]

                if (t >= t_init/dt) & (t <= t_end/dt):
                    neut[compart-1] = rho
                else:
                    neut[compart-1] = 0
        else:
            for i in range(len(neut)): 
                neut[i] = 0
    
    return neut, stimuli_times

@njit
def solve_model_equations(dt = 0.01, sample_rate = 100, compartment_to_monitor = (0), t_total = 0, n_comparts = 1, 
                          connection_matrix = ((0,),), parameters = {0}, A = (0,), Vol = (0,), ratio_ER = (0,), stim_glu_types = ('none',), 
                          stim_glu_t_init = (0,), stim_glu_t_end = (0,), stim_glu_Hz = (1,), stim_glu_comparts = (0,), stim_DA_types = ('none',), 
                          stim_DA_t_init = (0,), stim_DA_t_end = (0,), stim_DA_Hz = (1,), stim_DA_comparts = (0,)):
    """Calculate the nummerical solution of the system of differential equations of
    the astrocyte model with the 4th-order Runge-Kutta method.

    Parameters
    ----------
    dt: float
        time step for the nummerical solution by the 4th-order Runge-Kutta method.
    sample_rate: integer
        sample rate for the intracellular calcium concentration output.
    compartment_to_monitor: list, tuple or numpy 1D-array 
        indicates which compartments to monitor the intracellular calcium 
        concentration.
    t_total: integer
        total simulation time (in seconds).
    n_comparts: integer
        number of compartments in the astrocyte compartmental model.
    connection_matrix: numpy 2D-array 
        compartment connections.
    parameters: numba dictionary 
        all model parameters.
    A: list, tuple or numpy 1D-array
        area of each compartment.
    Vol: list, tuple or numpy 1D-array 
        volume of each compartment
    ratio_ER: list, tuple or numpy 1D-array 
        cytosol-ER volume ratio of each compartment
    stim_glu_types: list or tuple 
        type of stimuli ("poisson", "constant" or "none") for the glutamatergic input.
    stim_glu_t_init: list, tuple or numpy 1D-array 
        initial time of glutamatergic stimulation.
    stim_glu_t_end: list, tuple or numpy 1D-array
        end time of glutamatergic stimulation.
    stim_glu_Hz: list, tuple or numpy 1D-array 
        frequency (in seconds) of glutamatergic input.
    stim_glu_comparts: list, tuple or numpy 1D-array 
        compartments under stimulation with the glutamatergic input.
    stim_DA_types: list or tuple
        type of stimuli ("poisson", "constant" or "none") for the dopaminergic input
    stim_DA_t_init: list, tuple or numpy 1D-array.
        initial time of dopaminergic stimulation 
    stim_DA_t_end: ist, tuple or numpy 1D-array 
        end time of dopaminergic stimulation.
    stim_DA_Hz: list, tuple or numpy 1D-array
        frequency (in seconds) of dopaminergic input.
    stim_DA_comparts: list, tuple or numpy 1D-array
        compartments under stimulation for the dopaminergic input.

    Return
    ------
    Ca_out: numpy array
        intracellular calcium concentration of the compartments given by the parameter
        compartment_to_monitor and with sample rate given by sample_rate
    """    
    # Number of iterations in the Runge-Kutta method        
    n_points = int(t_total/dt)
    
    # Save parameters and connection matrix
    p = parameters 
    M = connection_matrix
        
    # Model Output
    Ca_out = ones(shape=(len(compartment_to_monitor), int(n_points/sample_rate)))*p['Ca_irest']

    # Initial Values
    Ca_o  = ones(shape=(n_comparts))*p['Ca_orest']
    Na_o  = ones(shape=(n_comparts))*p['Na_orest'] 
    K_o   = ones(shape=(n_comparts))*p['K_orest']
    Ca_i  = ones(shape=(n_comparts))*p['Ca_irest']
    h     = ones(shape=(n_comparts))*p['h_rest'] 
    IP3   = ones(shape=(n_comparts))*p['IP3_rest']
    Ca_ER = ones(shape=(n_comparts))*p['Ca_ERrest']
    Na_i  = ones(shape=(n_comparts))*p['Na_irest']
    K_i   = ones(shape=(n_comparts))*p['K_irest']
    V     = ones(shape=(n_comparts))*p['v_rest']
    Glu_o = ones(shape=(n_comparts))*p['Glu_orest']
    DA_o  = ones(shape=(n_comparts))*p['DA_orest']

    # Stimuli
    stimuli_times_glu = zeros(n_comparts)
    stimuli_times_DA = zeros(n_comparts)

    ### Runge Kutta 4th Order ###
    for i_t in range(n_points):

        # Calculate neurotransmitter concentration and spike times
        Glu_o, stimuli_times_glu = calculate_stimuli(Glu_o, stimuli_times_glu, i_t, dt, stim_glu_types, stim_glu_t_init, stim_glu_t_end,
                                                    stim_glu_Hz, stim_glu_comparts, p['rho_glu'])
        DA_o, stimuli_times_DA  = calculate_stimuli(DA_o, stimuli_times_DA, i_t, dt, stim_DA_types, stim_DA_t_init, stim_DA_t_end, 
                                                    stim_DA_Hz, stim_DA_comparts, p['rho_DA'])

        # Runge-Kutta
        k1 = dt * model_eqs(M, p, A, Vol, ratio_ER, Ca_o, Na_o, K_o, Ca_i, h, IP3, Ca_ER, Na_i, K_i, V, Glu_o, DA_o)
        k2 = dt * model_eqs(M, p, A, Vol, ratio_ER, Ca_o + 0.5*k1[0], Na_o + 0.5*k1[1], K_o + 0.5*k1[2], Ca_i + 0.5*k1[3], 
                            h + 0.5*k1[4], IP3 + 0.5*k1[5], Ca_ER + 0.5*k1[6], Na_i + 0.5*k1[7], K_i + 0.5*k1[8], 
                            V + 0.5*k1[9], Glu_o + 0.5*k1[10], DA_o + 0.5*k1[11])
        
        k3 = dt * model_eqs(M, p, A, Vol, ratio_ER, Ca_o + 0.5*k2[0], Na_o + 0.5*k2[1], K_o + 0.5*k2[2], Ca_i + 0.5*k2[3], 
                            h + 0.5*k2[4], IP3 + 0.5*k2[5], Ca_ER + 0.5*k2[6], Na_i + 0.5*k2[7], K_i + 0.5*k2[8], 
                            V + 0.5*k2[9], Glu_o + 0.5*k2[10], DA_o + 0.5*k2[11])
        
        k4 = dt * model_eqs(M, p, A, Vol, ratio_ER, Ca_o + k3[0], Na_o + k3[1], K_o + k3[2], Ca_i + k3[3], h + k3[4], 
                            IP3 + k3[5], Ca_ER + k3[6], Na_i + k3[7], K_i + k3[8], V + k3[9], Glu_o + k3[10], 
                            DA_o + k3[11])
        
        # Update variables
        Ca_o  = Ca_o  + (k1[0 ] + 2*k2[0 ] + 2*k3[0 ] + k4[0 ])/6
        Na_o  = Na_o  + (k1[1 ] + 2*k2[1 ] + 2*k3[1 ] + k4[1 ])/6
        K_o   = K_o   + (k1[2 ] + 2*k2[2 ] + 2*k3[2 ] + k4[2 ])/6
        Ca_i  = Ca_i  + (k1[3 ] + 2*k2[3 ] + 2*k3[3 ] + k4[3 ])/6
        h     = h     + (k1[4 ] + 2*k2[4 ] + 2*k3[4 ] + k4[4 ])/6
        IP3   = IP3   + (k1[5 ] + 2*k2[5 ] + 2*k3[5 ] + k4[5 ])/6
        Ca_ER = Ca_ER + (k1[6 ] + 2*k2[6 ] + 2*k3[6 ] + k4[6 ])/6
        Na_i  = Na_i  + (k1[7 ] + 2*k2[7 ] + 2*k3[7 ] + k4[7 ])/6
        K_i   = K_i   + (k1[8 ] + 2*k2[8 ] + 2*k3[8 ] + k4[8 ])/6
        V     = V     + (k1[9 ] + 2*k2[9 ] + 2*k3[9 ] + k4[9 ])/6
        Glu_o = Glu_o + (k1[10] + 2*k2[10] + 2*k3[10] + k4[10])/6
        DA_o  = DA_o  + (k1[11] + 2*k2[11] + 2*k3[11] + k4[11])/6
        
        # Output
        if i_t % sample_rate == 0:
            for i, compart in enumerate(compartment_to_monitor):
                Ca_out[i, int(i_t/sample_rate)] = Ca_i[compart - 1]
        
    return Ca_out
