# ----------------------------------------------------------------------------
# Contributors: Thiago O. Bezerra
#               Antonio C. Roque
# ----------------------------------------------------------------------------
# File description:
#
# Run all simulations specified in text files from the Input directory.
# ----------------------------------------------------------------------------

from time import perf_counter

import matplotlib.pyplot as plt
import run_model
import os
import parameters

file_list = os.listdir(r'Input/')
output_list = os.listdir(r'Output/')

model_parameters = parameters.define_parameters()
error_files = [] # stores the file for which the simulation returned an error

print('Executed files:')

start_time = perf_counter()

for i_file, file in enumerate(file_list):
   plt.ioff()
   file_result_name = file[:-3] + 'csv'
	
   if file_result_name not in output_list:
      try:
         print(file)
         run_model.simulate_from_file(os.path.join('Input/',file), model_parameters = model_parameters)
      except:
         error_files.append(file) 

print(f'tempo = {perf_counter() - start_time}')