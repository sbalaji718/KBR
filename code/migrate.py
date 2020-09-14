import multiprocessing as mp
import subprocess
import concurrent.futures
import os
import numpy as np
from integrations import long_integration, short_integration, kozai_check


# make function needed for multiprocessing
def do_integration(intN):

        """ when running this code, change the parameters of long_integration"""
        # first do long integration 
        # arguments are i, minA, maxA, minE, maxE, maxI, totalTime (exponent), Nparticles, max_exit_distance
        # returns minA, maxA, minE, maxE, maxI, Nparticles, totalTime, filename
        amin = 38.81
        amax = 39.95
        emin = 0.0
        emax = 0.6
        imax = 35.0
        maxD = 85
        nParticles = 500
        totTime = 7
        
        
        #amin, amax, emin, emax, imax, maxD, nParticles, totTime, filename = long_integration(intN, amin, amax, emin, emax, imax, totTime, nParticles, maxD)

        #now do short integration
        # arguments:i, minA, maxA, minE, maxE, maxI, shortTime, fileName, snapshotSlice (0 = time 0, -2 = second to last, -1 = last)     
        # return sim right now
        # we want to do 3 short integrations starting with time 0, 10%totalTime, and totalTime
        
        filename = 'Sep052020.18.32_part500_time10000000_A_38.810-39.950_Q_15.524-63.920_I_0-0.611_E_0.000-0.600_even_q_0'.format(intN)
        shortSim0 = short_integration(nParticles, totTime, intN, amin, amax, emin, emax, imax, maxD, 1e5, filename, 0)
        shortSim1 = short_integration(nParticles, totTime, intN, amin, amax, emin, emax, imax, maxD, 1e5, filename, -2)
        shortSim2 = short_integration(nParticles, totTime, intN, amin, amax, emin, emax, imax, maxD, 1e5, filename, -1)
       
        #kozaiShortInt1 = kozai_check(nParticles, totTime, intN, amin, amax, emin, emax, imax, maxD, 5e7, filename, 0)
        #kozaiShortInt2 = kozai_check(nParticles, totTime, intN, amin, amax, emin, emax, imax, maxD, 5e7, filename, -1) 
        #kozaiShortInt3 = kozai_check(nParticles, totTime, intN, amin, amax, emin, emax, imax, maxD, 5e7, filename, -2)
        
        return filename


#multiprocessing 
#when we run this code, change this number. It determines how many parallel processes are happening
lenInt = 2

args = np.arange(lenInt)

with mp.Pool(lenInt) as pool:
    results = pool.map(do_integration,args)

