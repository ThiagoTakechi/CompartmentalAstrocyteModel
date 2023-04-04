from numba import njit 

@njit
def define_parameters():
    """Create a Python dictionaruy and return all model's parameters."""
    
    p = {}

    ### Physical Constants ###
    p['T'] = 303.16              # K
    p['F'] = 96500               # C/mole
    p['R'] = 8.314               # J/(mole * K)


    ### Resting State ###
    p['Ca_irest']  = 7.3e-5         # mmole/m**3
    p['Ca_ERrest'] = 21.94786136e-3 # mmole/m**3
    p['Ca_orest']  = 1.8            # mmole/m**3
    p['Na_irest']  = 15             # mmole/m**3
    p['Na_orest']  = 145            # mmole/m**3
    p['K_irest']   = 100            # mmole/m**3
    p['K_orest']   = 3              # mmole/m**3
    p['IP3_rest']  = 0.1917e-3      # mmole/m**3
    p['h_rest']    = 0.802861638    # 1
    p['v_rest']    = -85e-3         # volt
    p['Glu_orest'] = 0              # mmole/m**3
    p['DA_orest']  = 0              # mmole/m**3


    ### IP3 Dynamics ###
    # PLC B Synthesis
    p['K_p']  = 10e-3            # mmole/m**3
    p['K_pi'] = 0.6e-3           # mmole/m**3

    # IP3 PLC Delta Synthesis
    p['v_delta']     = 0.025e-3  # mmole/m**3 * s
    p['kappa_delta'] = 1.5e-3    # mmole/m**3
    p['K_PLCdelta']  = 0.1e-3    # mmole/m**3

    # IP3-3K Degradation
    p['v_3K']        = 2e-3      # mmole/m**3 / s
    p['K_D']         = 0.7e-3    # mmole/m**3
    p['K_3']         = 1e-3      # mmole/m**3

    # IP-5P Degradation
    p['r_5P']    = 0.04      # 1/s


    ### Glutamate Transmission ###
    p['rho_glu']    = 0.5e-3     # mmole/m**3
    p['G_glu']      = 100        # 1/s
    p['K_R']        = 1.3e-3     # mmole/m**3
    p['v_beta']     = 0.674e-3   # mmole/m**3 * s
    p['alpha']      = 0.7


    ### Dopamine Transmission ###
    p['rho_DA']     = 3e-3       # mmole/m**3  
    p['G_DA']       = 4.201      # 1/s
    p['v_DA']       = 2.5e-5     # mmole/m**3 * s
    p['K_DA']       = 5e-3       # mmole/m**3
    p['beta']       = 0.5


    ### Ca ER Leak Current ###
    p['r_L']  = 0.11             # 1/s


    ### SERCA Current ###
    p['v_ER'] = 11.93e-3         # mmole/m**3 * s
    p['K_ER'] = 0.1e-3           # mmole/m**3


    ### h Dynamics ###
    p['d_1'] = 0.13e-3           # mmole/m**3
    p['d_5'] = 0.08234e-3        # mmole/m**3
    p['d_2'] = 1.049e-3          # mmole/m**3
    p['d_3'] = 0.9434e-3         # mmole/m**3
    p['a_2'] = 0.2               # m**3/(mmole * s)
    p['r_C'] = 6                 # 1/s


    ### GluT ###
    p['J_GluTmax'] = 0.68        # amp / metre ** 2
    p['K_GluTmN']  = 15          # mmole / m**3
    p['K_GluTmK']  = 5           # mmole / m**3
    p['K_GluTmg']  = 34e-3       # mmole / m**3


    ### NKA ###
    p['J_NKAmax'] = 1.52         # amp / metre ** 2
    p['K_NKAmN'] = 10            # mmole/m**3
    p['K_NKAmK'] = 1.5           # mmole/m**3

    ### NCX ###
    p['J_NCXmax'] = 0.0001          # amp/metre**2
    p['K_NCXmN'] = 87.5          # mmole/m**3
    p['K_NCXmC'] = 1.380         # mmole/m**3
    p['k_sat'] = 0.1             # 1
    p['eta'] = 0.35              # 1


    ### Voltage Parameter ###
    p['C_m'] = 1.0e-2            #F/m**2 


    ### Leakage Currents ###
    p['g_Naleak'] = 13.482808     # S/m**2
    p['E_Na']     = 61e-3         # V
    p['g_Kleak']  = 145.814171    # S/m**2
    p['E_K']      = -94e-3        # V
    
    
    ### Diffusion Constants ###
    p['D_Ca']   = 0.2          # m**2/s
    p['D_CaER'] = 0.001          # m**2/s
    p['D_IP3']  = 0.2          # m**2/s
    p['D_Na']   = 0.316        # m**2/s
    p['D_K']    = 0.938        # m**2/s
    p['D_Cao']  = 4.52         # m**2/s
    p['D_Nao']  = 26.6        # m**2/s
    p['D_Ko']   = 1.732        # m**2/s
    p['D_glu']  = 4e-4        # m**2/s
    p['D_DA']   = 13.8         # m**2/s
    
    return p