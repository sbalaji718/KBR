import multiprocessing as mp
import numpy as np
import subprocess
import concurrent.futures
import os
from integrations import long_integration, short_integration, check_resonance_make_plots, kozai_integration, check_kozai_make_plots


# make function needed for multiprocessing
def do_integration(intN):

    """ when running this code, change the parameters of long_integration"""
    # first do long integration 
    # arguments are int_count, minA, maxA, minE, maxE, integration_times, Nparticles, long_int_file= False
    # returns sim, filename

    minA = 38.81
    maxA = 40.0
    minE = 0.0
    maxE = 0.6
    Nparticles = 70
    timearr = np.array([0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 2e9, 3e9, 4.5e9])

    # let's say you already have a long integration. Then the last argument is long_int_file = filename    
    
    filename = "Jan232021.07.15_part70_time4500000000.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_{}".format(intN) 

    longsim, filename = long_integration(intN, minA, maxA, minE, maxE, timearr, Nparticles,  long_int_file = filename)

    #now do short integration
    # arguments:int_count, simarchive, sim_length, indexSimulation, filename    
    # return sim right now
    # we want to do 3 short integrations starting with time 0, 10%totalTime, and totalTime

#    shortSim0 = short_integration(intN, longsim, 1e5, 0, filename)
#    checkres0 = check_resonance_make_plots(shortSim0)

    shortSim = short_integration(intN, longsim, 1e5, -7, filename)
    checkres = check_resonance_make_plots(shortSim)

 #   shortSim1 = short_integration(intN, longsim, 1e5, -6, filename)
  #  checkres1 = check_resonance_make_plots(shortSim1)

#    shortSim2 = short_integration(intN, longsim, 1e5, -5, filename)
#    checkres2 = check_resonance_make_plots(shortSim2)

#    shortSim3 = short_integration(intN, longsim, 1e5, -4, filename)
#    checkres3 = check_resonance_make_plots(shortSim3)

#    shortSim4 = short_integration(intN, longsim, 1e5, -1, filename)
#    checkres4 = check_resonance_make_plots(shortSim4)


    shortSim0 = "Jan232021.07.15_part70_time4500000000.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_{}_short+100000.0.bin_data_array_100000.00000133288".format(intN)
    shortKozai0 = kozai_integration(intN, longsim, 5e7, 0, filename)
    checkKozai0 = check_kozai_make_plots(shortSim0, shortKozai0)

#    shortSim1 = "Jan232021.07.15_part70_time4500000000.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_{}_short+100000.0.bin_data_array_10100000.38075692.csv".format(intN)
#    shortKozai1 = kozai_integration(intN, longsim, 5e7, -6, filename)
#    checkKozai1 = check_kozai_make_plots(shortSim1, shortKozai1)

#    shortSim2 = "Jan232021.07.15_part70_time4500000000.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_{}_short+100000.0.bin_data_array_100100000.33945726.csv".format(intN)
#    shortKozai2 =  kozai_integration(intN, longsim, 5e7,-5, filename)
#    checkKozai2 = check_kozai_make_plots(shortSim2, shortKozai2)

#    shortSim3 = "Jan232021.07.15_part70_time4500000000.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_{}_short+100000.0.bin_data_array_1000100000.1482611.csv".format(intN)
#    shortKozai3 = kozai_integration(intN, longsim, 5e7, -4, filename)
#    checkKozai3 = check_kozai_make_plots(shortSim3, shortKozai3)


#    shortSim4 = "Jan232021.07.15_part70_time4500000000.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_{}_short+100000.0.bin_data_array_4500100000.219088.csv".format(intN)
#    shortKozai4 = kozai_integration(intN, longsim, 5e7, -1, filename)
#    checkKozai4 = check_kozai_make_plots(shortSim4,shortKozai4)


    return filename


#multiprocessing 
#when we run this code, change this number. It determines how many parallel processes are happening
lenInt = 39 # this one was 39

args = np.arange(lenInt)

with mp.Pool(lenInt) as pool:
    results = pool.map(do_integration, args)

