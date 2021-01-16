import multiprocessing as mp
import numpy as np
import subprocess
import concurrent.futures
import os
from integrations import long_integration, short_integration, check_resonance_make_plots#, kozai_check, integrate_3_5bill_more


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
    Nparticles = 100
    timearr = np.array([0, 1e5, 1e6])

    # let's say you already have a long integration. Then the last argument is long_int_file = filename    
    # filename = "Jan082021.01.28_part10_time100.0_A_38.810-40.000_iSig_14_E_0.000-0.600_even_q_0"

    longsim, filename = long_integration(intN, minA, maxA, minE, maxE, timearr, Nparticles) #, long_int_file = filename)

    #now do short integration
    # arguments:int_count, simarchive, sim_length, indexSimulation, filename    
    # return sim right now
    # we want to do 3 short integrations starting with time 0, 10%totalTime, and totalTime

    shortSim0 = short_integration(intN, longsim, 1e5, 0, filename)
    #checkres0 = check_resonance_make_plots(shortSim0)
    shortSim1 = short_integration(intN, longsim, 1e5, 1, filename)
    # checkres1 = check_resonance_make_plots(shortSim1)
    shortSim2 = short_integration(intN, longsim, 1e5, -1, filename)
    # checkres2 = check_resonance_make_plots(shortSim2)


    shortKozai = short_integration(intN, longsim, 5e7, 0, filename)
    # checkKozai = check_kozai()
    shortKozai = short_integration(intN, longsim, 5e7, 1, filename)
    shortKozai = short_integration(intN, longsim, 5e7, -1, filename)
    


    return filename


#multiprocessing 
#when we run this code, change this number. It determines how many parallel processes are happening
lenInt = 1

args = np.arange(lenInt)

with mp.Pool(lenInt) as pool:
    results = pool.map(do_integration, args)

