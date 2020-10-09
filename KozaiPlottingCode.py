import rebound
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import csv
import pdb
import glob
import timeit
import time
import os as osp
import shutil
import itertools
import pandas as pd


start_time = time.time()


datDate = 'Oct062020.17.55'

Nparticles = 100           #Number of test particles
Nout = 1000                 #Number of integrations
Ntotal = Nparticles + 5  #Number of test particles + giant planets

minA = 38.81
maxA = 39.95
minE = float(0.)                   #minimum eccentricity argument
maxE = float(0.6)                   #maximum eccentricity argument
minQ = minA*(1.-maxE)          #Perihelion distance
maxQ = maxA*(1.+maxE)           #Apehelion distance
maxI = np.round(35,3)                #maximum inclination argument
totalTime = int(1e7)
short_time = 5e7
intN= 0

fileName = '{}_part{}_time{}_A_{:.3f}-{:.3f}_Q_{:.3f}-{:.3f}_I_{:.3f}_E_{:.3f}-{:.3f}_even_q_{}'.format(datDate,Nparticles,totalTime,minA,maxA,minQ,maxQ,maxI,minE,maxE,intN)


#making this a function instead to be able to call it for the different plots we want to make

def create_kozai_plots(dat, ST, totalExpTime, integrationN, minA, maxA, minE, maxE, imax, shortTime, fileName, indexSimulation):
    """
    dat is the date of long integration that is to be used in Mmmddyyyy.HH.MM format
    ST is the start time of short integration 
    """
    
    master_dir = '{}_part{}_time{}_A_{:.3f}-{:.3f}_Q_{:.3f}-{:.3f}_I_{:.3f}_E_{:.3f}-{:.3f}_even_q'.format(dat,Nparticles,totalTime,minA,maxA,minQ,maxQ,imax,minE,maxE)
    #print(master_dir)
    destinPath = '/data/galadriel/Sricharan/KBO/KBR/Long_Integrations/{}/'.format(master_dir)
    #print(destinPath)
    #can get a list of the directory names without having to write them all down using glob
    # only using dat to be what we use to pull the name. can change this is we want
    fileDirectories = glob.glob("/data/galadriel/Sricharan/KBO/KBR/Long_Integrations/{}/{}*".format(master_dir, dat))
    #print(fileDirectories)
 

    
    sTemp    = 'TemporaryDirectory_time_{}'.format(np.round(ST))
    iRes     = 'In_Resonance'
    nRes     = 'Not_In_Resonance'
    sInt     = 'Short_Integrations'
    kozFiles = 'Kozai_Resonance'
    iKoz     = 'In_Kozai_Resonance'
    nKoz     = 'Not_In_Kozai_Resonance'

    mainDir    = '{}{}/'.format(destinPath,fileName)
    dirName    = '{}{}/{}'.format(destinPath,fileName,sInt)
    subDirTemp = '{}{}/{}/{}'.format(destinPath,fileName,sInt,sTemp)
    irDir      = '{}{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,iRes)
    nrDir      = '{}{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,nRes)
    kozDir     = '{}{}/{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,iRes,kozFiles)
    irKoz      = '{}{}/{}/{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,iRes,kozFiles,iKoz)
    nrKoz      = '{}{}/{}/{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,iRes,kozFiles,nKoz)
    
     
    try:
        osp.mkdir(irKoz)
        print ('Directory',irKoz,'created.')
    except FileExistsError:
        print("Directory",irKoz,"already exists.")

    try:
        osp.mkdir(nrKoz)
        print ('Directory',nrKoz,'created.')
    except FileExistsError:
        print("Directory",nrKoz,"already exists.")
    
    #the following code should be set up on first use to locate and store your simulations

    

    subDirectoriesTemp = glob.glob( '{}{}*/{}/{}'.format(destinPath,dat,sInt,sTemp)) #now have list of directory names
    
    
    koz_index = glob.glob( '{}{}*/{}/{}/{}/{}/Particles_in_Kozai_{}.txt'.format(destinPath,dat,sInt,sTemp,iRes,kozFiles,ST))
    nonkoz_index = glob.glob( '{}{}*/{}/{}/{}/{}/Particles_not_in_Kozai_{}.txt'.format(destinPath,dat,sInt,sTemp,iRes,kozFiles,ST))
    
    koz_plot_data = glob.glob( '{}{}*/{}/{}/{}/{}/koz_plot_data.txt'.format(destinPath,dat,sInt,sTemp,iRes,kozFiles))
    nonkoz_plot_data = glob.glob( '{}{}*/{}/{}/{}/{}/nonkoz_plot_data.txt'.format(destinPath,dat,sInt,sTemp,iRes,kozFiles))
    
    
    omega_data = glob.glob( '{}{}*/{}/{}/{}/{}/omega_koz_data.txt'.format(destinPath,dat,sInt,sTemp,iRes,kozFiles))
    nomega_data = glob.glob( '{}{}*/{}/{}/{}/{}/omega_nonkoz_data.txt'.format(destinPath,dat,sInt,sTemp,iRes,kozFiles))
    
    IT = 1e7
    ET = ST + IT 
    Nout = 1000
        
    t = np.linspace(ST, ET, Nout) 
    
    
    #Libration plots for particles in Kozai
    for i in range(len(omega_data)):
        content = np.genfromtxt('{}'.format(omega_data[i]))    
        kozai_ind_list = np.genfromtxt('{}'.format(koz_index[i]))
        kozai_ind_list = kozai_ind_list.astype(dtype = np.int32)
        for j in range(len(content)):
            plt.figure(figsize=(15,10))
            plt.title('Particle {} in Kozai'.format(kozai_ind_list[j]), fontsize = 24)
            plt.xlabel('Time(years)', fontsize = 18)
            plt.ylabel('argument of pericenter (omega)')
            plt.scatter(t,content[j]*180/np.pi,marker = '.', s = 10)
            plt.ylim(0,360)
            plt.savefig('{}/Particle {} omega vs Time Plot.png'.format(irKoz,kozai_ind_list[j])) 
    
    #Libration plots for particles not in Kozai
    for i in range(len(nomega_data)):
        ncontent = np.genfromtxt('{}'.format(nomega_data[i]))    
        nkozai_ind_list = np.genfromtxt('{}'.format(nonkoz_index[i]))
        nkozai_ind_list = nkozai_ind_list.astype(dtype = np.int32)
        for j in range(len(ncontent)):
            plt.figure(figsize=(15,10))
            plt.title('Particle {} in Kozai'.format(nkozai_ind_list[j]), fontsize = 24)
            plt.xlabel('Time(years)', fontsize = 18)
            plt.ylabel('argument of pericenter (omega)')
            plt.scatter(t,ncontent[j]*180/np.pi,marker = '.', s = 10)
            plt.xlim(ST, )
            plt.ylim(0,360)
            plt.savefig('{}/Particle {} omega vs Time Plot.png'.format(nrKoz,nkozai_ind_list[j])) 
    
    
    
    
    
    
    
    """
    for i in range(len(libplotArr)):
        plt.figure(figsize=(15,10))
        plt.title('Particle in Kozai', fontsize = 24)
        plt.xlabel('Time(years)', fontsize = 18)
        plt.ylabel('argument of pericenter (omega)')
        plt.scatter(t,libplotArr[i]*180/np.pi,marker = '.', s = 10)
        plt.ylim(0,360)
        plt.savefig('{}/Particle {} omega vs Time Plot.png'.format(irKoz,i))
    
    
    
    
    
    
    
    
    
    
    libplotArr = []  
    nlibplotArr = []  
    timeArr = []
    
    
    #This code doesn't work when there's only 1 particle in Kozai
    for omeg in omega_dat:
        #print(omeg)
        omegcontent = np.genfromtxt('{}'.format(omeg))
        #print(len(omegcontent))
        omegcon = np.array(omegcontent)
        #print(len(omegcon))
    

    
        for w in omegcon:
            libplotArr.append(w%(2*np.pi))
        #print(libplotArr)
        #print(len(libplotArr))
        #libplotArr = np.concatenate(libplotArr)  
 
    
    
    for nomeg in nomega_dat:
        nomegcontent = np.genfromtxt('{}'.format(nomeg))
        nomegcon = np.array(nomegcontent)
 
    for w in nomegcon:
        #print(w)
        nlibplotArr.append(w%(2*np.pi))
    #nlibplotArr = np.concatenate(nlibplotArr)   
    
    t = np.linspace(ST, ET, Nout)  
    
    #print(len(t))
    #print(len(libplotArr))
    
    for i in range(len(libplotArr)):
        plt.figure(figsize=(15,10))
        plt.title('Particle in Kozai', fontsize = 24)
        plt.xlabel('Time(years)', fontsize = 18)
        plt.ylabel('argument of pericenter (omega)')
        plt.scatter(t,libplotArr[i]*180/np.pi,marker = '.', s = 10)
        plt.ylim(0,360)
        plt.savefig('{}/Particle {} omega vs Time Plot.png'.format(irKoz,i))



    for i in range(len(nlibplotArr)):
        plt.figure(figsize=(15,10))
        plt.title('Particle not in Kozai', fontsize = 24)
        plt.xlabel('Time(years)', fontsize = 18)
        plt.ylabel('argument of pericenter (omega)')
        plt.scatter(t,nlibplotArr*180/np.pi,marker = '.', s = 10)
        plt.ylim(0,360)
        plt.savefig('{}/Particle {} omega vs Time Plot.png'.format(nrKoz,i))
    
    plottingArr = []
    nplottingArr = []

    for j in range(2):
        plottingArr.append([])
        nplottingArr.append([])
    
    
    
    for kozDat in koz_plot_dat:
        #print(koz_plot_dat)
        content = np.genfromtxt('{}'.format(kozDat))
        #print(content)
        con0 = np.array([content[0]])
        con1 = np.array([content[1]])
        #print(con1)
        
        for e in con0:
            #print("e", e)
            plottingArr[0].append(e) # eccentricity
        for i in con1:
            plottingArr[1].append(i) # inclination

    
    print("plottinArr before flatten", plottingArr[0])
    
    
    print(plottingArr)
    print(len(plottingArr[0]))
    # had to do this because was getting error for the case where there was only 1 resonant particle
    # error was happening because can't look through an array of 1 value
    # concatenate make the list of arrays into one array
    #plottingArr[0] = np.concatenate(plottingArr[0])
    #plottingArr[1] = np.concatenate(plottingArr[1])
    

    #print("plottingArr0", plottingArr[0])

    #Plotting array with nonresonant particles split into 3 sublists each corresponding to an orbital parameter necessary for plotting
    #semi-major axis values

    for nonkozDat in nonkoz_plot_dat:
        #print(nonresDat)
        ncontent = np.genfromtxt('{}'.format(nonkozDat))
        ncon0 = np.array([ncontent[0]])
        ncon1 = np.array([ncontent[1]])

        for e, i in zip(ncon0, ncon1):
            #print("a", a)
            #print("e", e)
            nplottingArr[0].append(e) # eccentricity
            nplottingArr[1].append(i) # inclination
            

    #print("n plotting before flatten", nplottingArr[0])

    nplottingArr[0] = np.concatenate(nplottingArr[0])
    nplottingArr[1] = np.concatenate(nplottingArr[1])
    

    

    print(plottingArr)
    print(nplottingArr)
    
    plotDestinPath = '/data/galadriel/Sricharan/Plotting_files/'.format(dat)
    plotDatePath = '{}{}'.format(plotDestinPath, dat)

    try: 
        osp.mkdir(plotDestinPath)
        print('Directory',plotDestinPath,'created.')
    except FileExistsError: 
        print('Directory',plotDestinPath,'already exists.') 

    try: 
        osp.mkdir(plotDatePath)
        print('Directory',plotDatePath,'created.')
    except FileExistsError: 
        print('Directory',plotDatePath,'already exists.') 

    
    #eccentricity vs semi major axis t = 1e7
    fig = plt.figure(figsize=(10,7))
    ax1 = plt.subplot(111)
    #plt.xlim([37,42])
    #plt.ylim([0, 1.1])
    plt.title('e vs. i, t = {:e}'.format(ST), fontsize = 24)
    plt.xlabel('eccentricity', fontsize = 18)
    plt.ylabel('inclination', fontsize = 18)
    ax1.scatter(nplottingArr[0],nplottingArr[1],marker = '.', c='r', label = 'nonkozai particles')
    ax1.scatter(plottingArr[0],plottingArr[1],marker = '.', c='b', label = 'kozai particles')

    #plt.plot(x19,eccentricity_from_peri_19, color = 'darkorange', linestyle = 'dashed', label = "p = 19AU")
    #plt.plot(x24,eccentricity_from_peri_24, color = 'm', linestyle = 'dashed', label = "p=23AU")
    #plt.plot(x29,eccentricity_from_peri_29, color = 'forestgreen', linestyle = 'dashed', label = "p=29AU")
    #plt.plot(x34,eccentricity_from_peri_34, color = 'grey', linestyle = 'dashed', label = "p=34AU")
    #plt.plot(x37,eccentricity_from_peri_37, color = 'blueviolet', linestyle = 'dashed', label = "p=37AU")

    #plt.axvline(x=minA, color = 'black', linestyle = '-')
    #plt.axvline(x=maxA, color = 'black', linestyle = '-')
    #plt.axhline(y=0.6, color = 'black', linestyle = '-')                       #some particles go slightly past 0.6      
    #plt.legend(loc = 'upper right', prop={'size': 10})
    plt.show()
    plt.savefig('/data/galadriel/Sricharan/Plotting_files/{}/evi_t={}.png'.format(dat, ST))
    plt.clf()
    """
    return ST
    
    



#create_kozai_plots(datDate, 0.0)
#create_kozai_plots(datDate, 1e6, totTime, intN ,minA, maxA, minE, maxE, maxI, short_time, fileName, -1)
create_kozai_plots(datDate, 1e7, totalTime, intN ,minA, maxA, minE, maxE, maxI, short_time, fileName, -1)


