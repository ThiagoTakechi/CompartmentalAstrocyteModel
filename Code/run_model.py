# ----------------------------------------------------------------------------
# Contributors: Thiago O. Bezerra
#               Antonio C. Roque
# ----------------------------------------------------------------------------
# File description:
#
# Functions to run the astrocyte model from text file, load astrocyte morphology 
# from text file in swc format, make and save a graph with the simulation output
# (intracellular calcium concentration), and analyze and save output data.
# ----------------------------------------------------------------------------

from warnings import warn

from numpy import array, zeros, pi, sqrt, exp

from numba import types 
from numba.typed import Dict

import os
import csv

import astrocyte_model


def simulate_from_file(filename, model_parameters):
    """Run a simulation of the compartimental astrocyte model with parameters defined
    from a text file (.txt). Each parameter identifier must be separated by a space 
    charecter.
    
    The text file format must contain (simulation_parameter): the morphology file path
    (morphology_filepath), the step for nummerical integration (dt), the sample rate
    for the output (sample_rate), the total simulation time (t_total), the compartments
    to monitor (compartment_to_monitor), whether or not to save the result as a figure
    (save_fig) and the figure file path (fig_path), whether or not the variables start
    values should be calculated (init_var) to impose equilibrium at the beginning and 
    whether or not the output should be analyzed (analyze_data).
    
    If the type of glutamatergic stimulation (stim_glu) is different from "none", the 
    text file also must contain the compartments under glutamatergic stimulation 
    (stimulated_compartments_glu), the type of glutamatergic stimulation (stim_glu), the
    frequency of glutamatergic stimulation (Hz_glu), the starting and ending time for the 
    glutamatergic stimulation (t_iglu and t_eglu).
    
    If the type of dopaminergic stimulation (stim_DA) is different from "none", the 
    text file also must contain the compartments under dopaminergic stimulation 
    (stimulated_compartments_DA), the type of dopaminergic stimulation (stim_DA), the 
    frequency of dopaminergic stimulation (Hz_DA), the starting and ending time for the 
    dopaminergic stimulation (t_iDA and t_eDA).

    If neither save_fig nor analyze_data were requested, there will be no output from
    this function. save_fig, analyze_data and init_var must be a character "Y" (yes)
    or it will not save the figure, analyze the data or initialize the variables. The
    parameter thresh (in mmol) configures the minimum height (peak amplitude) of the 
    calcium concentration to count as a calcium signal.

    To change the model's parameters, the line must beggin with "change_param"
    followed by the parameter name (as specified in the parameter.py file) and
    the new value.

    Example of input text file to simulate 100 seconds of an astrocyte with 17 
    compartments, analyze the data, not save a figure, only dopaminergic stimulation:

    simulation_parameter init_var Y
    simulation_parameter morphology_filepath ../Morphology/Bifurcation_17comparts.txt
    simulation_parameter save_fig N
    simulation_parameter analyze_data Y
    simulation_parameter thresh 0.15e-3
    simulation_parameter dt 0.01e-3
    simulation_parameter sample_rate 1000
    simulation_parameter t_total 100
    simulation_parameter compartment_to_monitor 15,16,17
    stim_DA poisson 0 100 10 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
    stim_glu none
    change_param r_C 4

    Parameters
    ----------
    filename: str
        file path to the input text file configuring the simulation
    model_parameters: Python or Numba dictionary
        all model's parameters
    """
    
    ### Get simulation parameters from file ###
    # Dictionary to save the simulation parameters
    simulation_parameters = {}
    
    # Lists with the glutamatergic input parameters
    stim_glu_types = []
    stim_glu_t_init = []
    stim_glu_t_end = []
    stim_glu_Hz = []
    stim_glu_comparts = []

    # Lists with the dopaminergic input parameters
    stim_DA_types = []
    stim_DA_t_init = []
    stim_DA_t_end = []
    stim_DA_Hz = []
    stim_DA_comparts = []

    # Auxiliar dictionary for changing model's parameters if specified
    model_parameters_temp = {}

    with open(filename, 'r') as file:

        for line_n, line in enumerate(file):
            
            splitted = line.split()
            
            if len(splitted) < 2:
                raise ValueError((f"Each line of the file has to contain "
                                     f"2 or more space-separated entries, but line "
                                     f"{line_n + 1} contains {len(splitted)}."))
                
            
            # Save stimulation parameters
            if splitted[0] == 'simulation_parameter':

                simulation_parameters[splitted[1]] = splitted[2]

            # Save glutamatergic and dopaminergic stimuli parameters
            elif splitted[0] == 'stim_glu':

                if splitted[1] == 'poisson':
                    # type of stimulus, initial time, end time, frequency, comparts
                    stim_glu_types.append(str(splitted[1]))
                    stim_glu_t_init.append(float(splitted[2]))
                    stim_glu_t_end.append(float(splitted[3]))
                    stim_glu_Hz.append(float(splitted[4]))
                    stim_glu_comparts.append(array(splitted[5].split(','), dtype = int))
                    
                elif splitted[1] == 'constant':
                    # type of stimuli, initial time, end time, comparts
                    stim_glu_types.append(str(splitted[1]))
                    stim_glu_t_init.append(float(splitted[2]))
                    stim_glu_t_end.append(float(splitted[3]))
                    stim_glu_Hz.append(0) # lists cannot be empty
                    stim_glu_comparts.append(array(splitted[4].split(','), dtype = int))

                elif splitted[1] == 'none':
                    # lists cannot be empty
                    stim_glu_types.append(str(splitted[1]))
                    stim_glu_t_init.append(0)
                    stim_glu_t_end.append(0)
                    stim_glu_Hz.append(0)
                    stim_glu_comparts.append(zeros(1))
                else:
                    raise ValueError('Type of glutamate stimulation not recognized. The options are poisson, constant or none.')

            elif splitted[0] == 'stim_DA':

                if splitted[1] == 'poisson':
                    # type of stimulus, initial time, end time, frequency, comparts
                    stim_DA_types.append(str(splitted[1]))
                    stim_DA_t_init.append(float(splitted[2]))
                    stim_DA_t_end.append(float(splitted[3]))
                    stim_DA_Hz.append(float(splitted[4]))
                    stim_DA_comparts.append(array(splitted[5].split(','), dtype = int))
                    
                elif splitted[1] == 'constant':
                    # type of stimuli, initial time, end time, comparts
                    stim_DA_types.append(str(splitted[1]))
                    stim_DA_t_init.append(float(splitted[2]))
                    stim_DA_t_end.append(float(splitted[3]))
                    stim_DA_Hz.append(0) # lists cannot be empty
                    stim_DA_comparts.append(array(splitted[4].split(','), dtype = int))

                elif splitted[1] == 'none':
                    # lists cannot be empty
                    stim_DA_types.append(str(splitted[1]))
                    stim_DA_t_init.append(0)
                    stim_DA_t_end.append(0)
                    stim_DA_Hz.append(0)
                    stim_DA_comparts.append(zeros(1))
                else:
                    raise ValueError('Type of dopaminergic stimulation not recognized. The options are poisson, constant or none.')

            # Save new model parameters values if specified
            elif splitted[0] == 'change_param':
                model_parameters_temp[splitted[1]] = splitted[2]

    ### Change model parameters if specified by the change_param ###
    if model_parameters_temp:

        for key, value in model_parameters_temp.items():
            # aqui verificar
            model_parameters[key] = float(value)

    ### Calculate initial values if specified in the text file by the simulation 
    ### parameter init_var. Impose equilibrium at the beggining.
    if simulation_parameters['init_var'] == 'Y':
        model_parameters['IP3_rest']  = float(bisec_method(astrocyte_model.null_IP3, a = 0, b = 9e-3, p = model_parameters))
        model_parameters['h_rest']    = float(astrocyte_model.null_h(p = model_parameters))
        model_parameters['Ca_ERrest'] = float(astrocyte_model.null_ca_er(p = model_parameters))
        model_parameters['g_Naleak']  = float(astrocyte_model.null_na_i(p = model_parameters))
        model_parameters['g_Kleak']   = float(astrocyte_model.null_k_i(p = model_parameters))

    ### Get morphological parameters ###  
    points = read_from_swc(simulation_parameters['morphology_filepath'])
    morp_params = calculate_morphological_parameters(points)
    connection_matrix = build_connection_matrix(points)

    # Correct diffusion factor for compartment dimensions
    connection_matrix[0, 1] = 0.183e-4
    connection_matrix[1, 0] = 0.049

    n_comparts = morp_params.shape[0]
    A = morp_params[:,3]
    Vol = morp_params[:,5]
    ratio_ER = morp_params[:,6]
    
    ### Get simulation parameters ###
    dt = float(simulation_parameters['dt'])
    sample_rate = int(simulation_parameters['sample_rate'])
    t_total = float(simulation_parameters['t_total'])
    p = create_numba_dictionary(model_parameters)
    compartment_to_monitor = array(simulation_parameters['compartment_to_monitor'].split(','), dtype = int)
    
    ### Glutamate stimulation parameters ###
    # From lists to tuples
    stim_glu_types = tuple(stim_glu_types)
    stim_glu_t_init = tuple(stim_glu_t_init)
    stim_glu_t_end = tuple(stim_glu_t_end)
    stim_glu_Hz = tuple(stim_glu_Hz)
    stim_glu_comparts = tuple(stim_glu_comparts)
    
    ### Dopaminergic stimulation parameters ###
    # From lists to tuples
    stim_DA_types = tuple(stim_DA_types)
    stim_DA_t_init= tuple(stim_DA_t_init)
    stim_DA_t_end = tuple(stim_DA_t_end)
    stim_DA_Hz = tuple(stim_DA_Hz)
    stim_DA_comparts = tuple(stim_DA_comparts)
    
    ### Short simulation to test parameters and for JIT ###
    Ca_i = astrocyte_model.solve_model_equations(dt = dt, sample_rate = sample_rate, compartment_to_monitor = compartment_to_monitor, 
                                                t_total = 0.001, n_comparts = n_comparts, connection_matrix = connection_matrix, 
                                                parameters = p, A = A, Vol = Vol, ratio_ER = ratio_ER, stim_glu_types = stim_glu_types,
                                                stim_glu_t_init = stim_glu_t_init, stim_glu_t_end = stim_glu_t_end, stim_glu_Hz = stim_glu_Hz,
                                                stim_glu_comparts = stim_glu_comparts, stim_DA_types = stim_DA_types, 
                                                stim_DA_t_init = stim_DA_t_init, stim_DA_t_end = stim_DA_t_end, stim_DA_Hz = stim_DA_Hz, 
                                                stim_DA_comparts = stim_DA_comparts)
    
    ### Run simulation ###
    Ca_i = astrocyte_model.solve_model_equations(dt = dt, sample_rate = sample_rate, compartment_to_monitor = compartment_to_monitor, 
                                                t_total = t_total, n_comparts = n_comparts, connection_matrix = connection_matrix, 
                                                parameters = p, A = A, Vol = Vol, ratio_ER = ratio_ER, stim_glu_types = stim_glu_types,
                                                stim_glu_t_init = stim_glu_t_init, stim_glu_t_end = stim_glu_t_end, stim_glu_Hz = stim_glu_Hz,
                                                stim_glu_comparts = stim_glu_comparts, stim_DA_types = stim_DA_types, 
                                                stim_DA_t_init = stim_DA_t_init, stim_DA_t_end = stim_DA_t_end, stim_DA_Hz = stim_DA_Hz, 
                                                stim_DA_comparts = stim_DA_comparts)
    
    ### Make graph if specified in text file by the simulation parameter save_fig ###
    if simulation_parameters['save_fig'] == "Y":
        make_graph(ca_list = Ca_i, t_total = t_total, labels = compartment_to_monitor, fig_path = simulation_parameters['fig_path'],
                   fig_name = filename.split('/')[-1][:-4])

    ### Analyze and save analyzed data if specified in text file by the simulation parameter analyze_data ###
    if simulation_parameters['analyze_data'] == "Y":
        analyze_data(ca_list = Ca_i, thresh = float(simulation_parameters['thresh']), dt = dt, sample_rate = sample_rate, 
                    file_name = filename.split('/')[-1][:-4])

def make_graph(ca_list, t_total, labels, fig_path = '', fig_name = ''):
    """Make a graph (Ca_i x t) with the simulation output and save as a figure in
    pdf format.

    Parameters
    ----------
    ca_list: numpy 2D-array or list of 1D-arrays
        intracellular concentration for each monitored compartment.
    t_total: float
        total simulation time.
    labels: int
        index number of the monitored compartments.
    fig_path: sre
        folder in which to save the graph as a figure in pdf format.
    fig_name: str
        figure file name.
    """

    import matplotlib.pyplot as plt
    from numpy import linspace

    plt.ioff()
    fig, ax = plt.subplots(1, 1)

    t = linspace(0, t_total, ca_list.shape[1])

    for i, ca_i in enumerate(ca_list):
        ax.plot(t, ca_i*1000, label = labels[i])
    

    ax.set_xlabel('t (s)', fontsize=12)
    ax.set_ylabel(r'$\mathrm{[Ca^{2+}]_i~(\mu M)}$', fontsize=12)

    fig.legend(ncol=8, loc='upper center')

    plt.ioff()

    name = fig_name
    path = os.path.join(fig_path, fig_name) + '.pdf'
    fig.savefig(path, format='pdf', dpi = 600, bbox_inches='tight')

    plt.close(fig)

def analyze_data(ca_list, thresh, dt, sample_rate, file_name = ''):
    """Analyze data and save the result as a csv file. Save the number of calcium
    signal, the maximum amplitude of the signal, the minimum intracellular calcium 
    concentration and the event times.

    Parameters
    ----------
    ca_list: numpy 2D-array or list of 1D-arrays
        intracellular concentration for each monitored compartment.
    thresh: float
        threshold for counting a calcium signal
    dt: float
        time step for the nummerical solution.
    sample_rate: int
        the sample rate for the intracellular calcium concentration output.
    file_name: str
        output file name.
    """
    
    if not(os.path.exists("Output/")):
        warn(f'Folder Output does not exist. File will be save in the current directory.')
        path = f'{file_name}.csv'
    else:
        path = f'{os.path.join("Output/", file_name)}.csv'

    with open(path, 'w', newline = '') as csvfile:
        
        csv_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(('Compart', 'NCaSignal', 'Max', 'Min'))

        for i, ca_i in enumerate(ca_list):

            t_ca_signals, n_ca_signal = count_ca_signals(ca_i, thresh, dt, sample_rate)
            max_ca = ca_i.max()
            min_ca = ca_i.min()

            csv_writer.writerow((i+1, n_ca_signal, max_ca, min_ca, t_ca_signals))

def count_ca_signals(ca, thresh, dt, sample_rate):
    """Count the number of calcium signals in the temporal series given a threshold.

    Parameters
    ---------
    ca: numpy 1D-array
        intracellular calcium concentration (Ca_i)
    thresh: float
        threshold for counting a calcium signal
    dt: float
        time step for the nummerical solution.
    sample_rate: int
        sample rate of the intracellular calcium concentration output.
    """

    from scipy.signal import find_peaks
    peaks = find_peaks(ca, height = thresh)[0]*dt*sample_rate

    return peaks, len(peaks)

def read_from_swc(filename):
    """Get astrocyte morphology from a text file (.txt) in the swc format.

    Each line in the text file must contain the following data space-separated:
    index, compartment type, x coordinate, y coordinate z coordinate, radius,
    index of parent node. So, each line must contain seven entries.

    x coordinate, y coordinate, z coordinate and radius must be in m. For the
    astrocyte models, comparment type is 0 for soma and 1 for processes. The
    parent index for the somatic compartment must be -1.

    Example for a 9-comparment morphology:
    1 0 0 0 0 20e-6 -1
    2 1 21e-6 0 0 2e-6 1
    3 1 22e-6 0 0 2e-6 2
    4 1 23e-6 0 0 2e-6 3
    5 1 24e-6 0 0 1e-6 4
    6 1 25e-6 0 0 0.5e-6 5
    7 1 26e-6 0 0 0.25e-6 6
    8 1 27e-6 0 0 0.125e-6 7
    9 1 28e-6 0 0 0.0625e-6 8

    Parameters
    ----------
    filename: str
        path to the morphology text file

    Return
    ------
    points: list of tuples
        loaded morphology from the text file
    """
    with open(filename,'r') as f:
        points = []
        
        for line_n, line in enumerate(f):
            
            if line.startswith('#') or len(line) == 0:
                continue
                
            splitted = line.split()
            
            if len(splitted) != 7:
                raise ValueError((f"Each line of an SWC file has to contain "
                                 f"7 space-separated entries, but line "
                                 f"{line_n + 1} contains {len(splitted)}."))
                
            index, comp_type, x, y, z, radius, parent = splitted
            
            points.append((int(index),
                          int(comp_type), 
                          float(x), 
                          float(y),
                          float(z), 
                          float(radius), 
                          int(parent)))
            
        return points

def calculate_morphological_parameters(points):
    """Calculate morpjological parameters from a specified astrocyte morphology. The
    somatic compartment is a sphere and the other comparments are cylinders.

    Let r be the radius of the somatic compartments, so its surface area, cross sectional
    areav and volume are calculate as follow:
    Surface Area = 4*pi*r^2 
    Cross Sectional Area = 4*pi*r^2 (flux from inside to outside of the sphere)
    Volume = 4/3*pi*r^3

    For the cylindrical compartments with radius r, the lenth, surface area, cross 
    sectional area and volume are calculate as:
    Length = euclidean distance between two compartments
    Surface Area = 2*pi*r*length 
    Cross Sectional Area = pi*r^2
    Volume = pi*r^2*length

    The cytosol-endoplasmic reticulum volume ratio is calculated as (Patrushev et al.
    2013):
    ratio_ER 0.15*exp(-(0.073e-6 * Surface Area/Volume)**2.34) 
    
    References
    ----------
    Patrushev I, Gavrilov N, Turlapov V, Semyanov A. Subcellular location of
    astrocytic calcium stores favors extrasynaptic neuron–astrocyte communication.
    Cell Calcium. 2013; 54:343-9

    Parameters
    ----------
    points: list of list or tuple (n x 7),
        list with the morphologycal parameter of each compartment (in swc format).

    Return
    ------
    morp_params: numpy array.
        Each columm represent:
            0 - Compartment type (0 for soma and 1 for process)
            1 - Compartment radius
            2 - Compartment length (radius for soma)
            3 - Surface area
            4 - Cross sectional area (surface area for soma)
            5 - Volume
            6 - Ratio ER
    """
    n_compart = len(points)
    morph_params = zeros(shape=(n_compart, 7))
    
    for i, compart in enumerate(points):
                
        if compart[-1] == -1:
            morph_params[i][0] = 0                                                              # Compartment Type (0 for soma)
            morph_params[i][1] = compart[-2]                                                    # Radius
            morph_params[i][2] = compart[-2]                                                    # Lenght (equal the radius for soma)
            
            morph_params[i][3] = 4*pi*compart[-2]**2                                            # Sphere surface area
            morph_params[i][4] = 4*pi*compart[-2]**2                                            # Sphere cross section area
            morph_params[i][5] = 4/3*pi*compart[-2]**3                                          # Sphere volume
            
        else: 
            morph_params[i][0] = 1                                                              # Compartment Type
            morph_params[i][1] = compart[-2]                                                    # Radius
            
            # If the parent compartment is the soma, the lenght of the compartment is calculated considering the radius of the soma
            if compart[-1] == 1:
                morph_params[i][2] = sqrt(compart[2]**2 + compart[3]**2 + compart[4]**2) - points[0][-2]      # Length
            else:
                morph_params[i][2] = sqrt((compart[2] - points[compart[-1] - 1][2])**2 + 
                                            (compart[3] - points[compart[-1] - 1][3])**2 +
                                            (compart[4] - points[compart[-1] - 1][4])**2)       # Length
            
            morph_params[i][3] = 2*pi* morph_params[i][1]*morph_params[i][2]                    # Cylinder surface area
            morph_params[i][4] = pi*morph_params[i][1]**2                                       # Cylinder cross section area
            morph_params[i][5] = pi*compart[-2]**2*morph_params[i][2]                           # Cylinder volume
            
        morph_params[i][6] = 0.15*exp(-(0.073e-6 * morph_params[i][3]/morph_params[i][5])**2.34)  # Ratio ER
        
    return morph_params

def build_connection_matrix(points):
    """Construct the connection matrix representing the astrocyte compartments
    connections. Each entry indicates a connected (= 1) or disconnected (= 0)
    compartment.
    
    Parameters
    ----------
    points: list of list or tuple (n x 7),
        list with the morphologycal parameter of each compartment (in swc format).
    
    Return
    ------
    connection_matrix: numpy array (number of compartments x number of compartments)
        matrix indicating compartment connections
    """
    n = len(points)
    connection_matrix = zeros(shape=(n,n))
    
    for connection in points:
        if connection[-1] != -1:
            connection_matrix[connection[0] - 1,connection[-1] - 1] = 1
            connection_matrix[connection[-1] - 1,connection[0] - 1] = 1
        
    return connection_matrix

def create_numba_dictionary(dictionary):
    """Create the numba type dictionary. Native Python dictionaries are not 
    compatible with numba.

    Parameters
    ----------
    dictionary: Python dictionary.

    Return
    ------
    result_dict: Numba dictionary
    """
    result_dict = Dict.empty(key_type = types.unicode_type,
                                       value_type = types.float64)

    for key in dictionary.keys():
        result_dict[key] = dictionary[key]
        
    return result_dict

def bisec_method(f, a, b, tol = 1e-10, **kwargs):
    """Calculate the roots of the function f in the interval [a,b] with a tolerance
    tol using the bisection method.

    Parameters
    ----------
    f: function
        function for which the root will be calculated.
    a: float
        lower bound for the interval in which to search the function root.
    b: float
        upper bound for the interval in which to search the function root.
    tol: float
        error accepted for calculating the root.
    **kwargs: keyword-arguments for the function f.

    Return
    ------
    Approximated function root
    """
    
    while (b - a >= tol):
        if f(a, **kwargs)*f(0.5*(b + a), **kwargs) < 0:            
            b = 0.5*(b + a)
        else:
            a = 0.5*(b + a)
            
    return 0.5*(b + a)
