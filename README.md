# CompartmentalAstrocyteModel

This repository contains the code for the simulation of the astrocyte compartmental model by Bezerra & Roque, Dopamine facilitates the response to glutamatergic inputs in astrocyte cell models. PLOS Computational Biology. 2023 (DOI:).

The main code is divided into four files: astrocyte_model.py, parameters.py, run_model.py and run_tests.py. The first file contains all model's equations and functions for generating the input and numericaly solve the system of differential equations. The second file contains the model's parameters. The file run_model.py contains functions to import the astrocyte morphology file (in swc format, see the Astrocyte Morphology section below) and to run a simulation with configuration given by a text file (see the Input File section below). The file run_tests.py run the model with all input files inside a folder "Input" and save the corresponding outputs in a folder "Output".

With the run_model.py it is possible to run a simulation and request the program to save the results (intracellular calcium concentration) as a graph, to analyze the output (number of calcium signals, maximum and minimum intracellular calcium concentration, and the calcium signal times for each compartment), and save the analyzed data in a csv file.

## Astrocyte Morphology

The astrocyte morphology model file must be in the swc format. The somatic compartment is a spheric comparment while the remaining compartments are cylinders. Each line must contain seven space separated entries. The first column indicate the compartment index. The second one the type of compartment (1 for soma and 7 for astrocytic processes). The third to fifth columns the x, y and z coordinates (in m), respectively. The sixth the compartment radius (in m). The seventh column indicates the parent compartment. In the case of the somatic compartment, the last entry must be -1. In this model, each cylindrical compartment has unitary length. 

**Example:**  
1 1 0 0 0 20e-6 -1  
2 7 21e-6 0 0 2e-6 1  
3 7 22e-6 0 0 2e-6 2  
4 7 23e-6 0 0 2e-6 3  
5 7 24e-6 0 0 1e-6 4  
6 7 25e-6 0 0 0.5e-6 5  
7 7 26e-6 0 0 0.25e-6 6  
8 7 27e-6 0 0 0.125e-6 7  
9 7 28e-6 0 0 0.0625e-6 8  

## Input File

To simulate the model with the run_model.py file, it must be indicated the simulation configurations inside a text file. This input files must contain all configurations for the program to correctly simulate the astrocyte model. This text file must be formated as follows. Each line in the file indicates one configuration for the simulation. The fisrt column entries can be: simulation parameters (simulation_parameter), glutamatergic and dopaminergic stimuli (stim_glu and stim_DA, repectively) and, if necessary, parameter changing commands (change_param). 

### Simulation Parameters

To configure the simulation parameters, each line must be divided in three space separated entries. In this case, the lines must always beggin with "simulation_parameter". Next, it must indicate the total simulation time (t_total, in seconds, float), the sample rate (sample_rate, in time points, sample frequency = 1/(sample_rate*dt), integer), the time step dt (dt, in seconds, float) for the numerical solution with the 4th-order Runge-Kutta method, the path to the morphology file (morphology_filepath, string), and the compartment indices for which the intracellular calcium concentration will be stored (compartment_to_monitor, sequence of integers separated by a ",", string). It must also indicate whether the program should initialize all variables in order to impose stationary equilibrium at the beggining of the simulation (init_var, "Y" or "N", character), whether to save the simulation output (intracellular calcium concentration of the monitored compartments) as a graph (save_fig, "Y" or "N", character) and whether to analyze the data and save the result in a text file (analyze_data, "Y" or "N", character). If the "save_fig" is passed as "Y", the input text file must contain the path to save figure (fig_path, string). Also, if the "analyze_data" is passed as "Y", the input file must contain a threshold value for counting the calcium signals (thresh, in mM, float). Finally, the last column must indicate the value of each simulation parameter.

**Example:**  
simulation_parameter t_total 600  
simulation_parameter dt 0.01e-3  
simulation_parameter sample_rate 1000  
simulation_parameter morphology_filepath ../Morphology/Linear_9comparts.txt  
simulation_parameter compartment_to_monitor 1,2,3,4,5,6,7,8,9  
simulation_parameter init_var Y  
simulation_parameter save_fig Y  
simulation_parameter fig_path ../Tests  
simulation_parameter analyze_data Y  
simulation_parameter thresh 0.15e-3  

### Stimuli Parameters

To configure the glutamatergic and dopaminergic stimuli parameters, each line indicates an input with different configurations. Entries also must be space separated. Each line must start with "stim_glu" for the glutamatergic input or "stim_DA" for the dopaminergic input. Next, it must indicate whether the input follows a poisson process ("poisson"), a constant input ("constant") or no stimulation ("none"). For the Poisson input, the line must contain the starting stimulation time (float), the final stimulation time (float), the input frequency (float) and the compartments under stimulation (sequence of numbers separated by ",", string). For the constant input, the line must contain the starting stimulation time (float), the final stimulation time (float) and the compartments under stimulation (sequence of numbers separated by ",", string). In the absence of glutamatergic and/or dopaminergic stimulation, the input text file must contatin lines with "stim_glu none" and/or "stim_DA none".

**Example:**  
stim_glu poisson 0 600 10 1,2,3,4,5,6,7,8,9  
stim_glu constant 0 100 1,2,3  
stim_DA none  

### Changing Model Parameter 

This cnfiguration is optional. However, it can be useful for the simulation of the model using different parameter values. So, to change any model parameters, the input text file line start with "change_param" followed by the parameter's name (string) and its value (float). All paramaters' names can be found in the parameters.py file. 

**Example:**  
change_param rho_glu 0.5e-2  
change_param G_glu 150  

## Requirements

Python 3.6  
Numba 0.55.1  
Numpy 1.21.5  
Matplotlib 3.5.1  
Scipy 1.7.3  
