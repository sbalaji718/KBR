import multiprocessing as mp
import numpy as np
import subprocess
import concurrent.futures
import os
from integrations import long_integration, short_integration, check_resonance#, kozai_check, integrate_3_5bill_more


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
    # maxD = 10
    Nparticles = 10
    timearr = np.array([0, 1e1, 1e2, 1e4])    

    longsim, filename = long_integration(intN, minA, maxA, minE, maxE, timearr, Nparticles)

    #now do short integration
    # arguments:int_count, simarchive, sim_length, indexSimulation, filename    
    # return sim right now
    # we want to do 3 short integrations starting with time 0, 10%totalTime, and totalTime

    shortSim0 = short_integration(intN, longsim, 100, 0, filename)
    checkres = check_resonance(shortSim0, Nparticles, 1000)


    return filename


#multiprocessing 
#when we run this code, change this number. It determines how many parallel processes are happening
lenInt = 1

args = np.arange(lenInt)

with mp.Pool(lenInt) as pool:
    results = pool.map(do_integration, args)

