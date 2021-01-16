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


start_time = time.time()


#maybe these will become the arguments for function, haven't decided 


base = 30.*(3./2.)**(2./3.) #~39AU
#print(base)

b    = 1.2                         #changed the range of a to be larger (from +- 0.5 to +- 2.5)
#minA = base - b                    #minimum semi-major axis distance     
#maxA = base + b                    #maximum semi-major axis distance
minA = 38.81
maxA = 39.95
minE = float(0.)                   #minimum eccentricity argument
maxE = float(0.6)                   #maximum eccentricity argument
minQ = (minA*(1.-maxE))            #Perihelion distance
maxQ = (maxA*(1.+maxE))            #Apehelion distance
maxI = 0.611                 #maximum inclination argument


#making this a function instead to be able to call it for the different plots we want to make

def create_plots(dat, ST):
    """
    dat is the date of long integration that is to be used in Mmmddyyyy.HH.MM format
    ST is the start time of short integration 
    """


    #can get a list of the directory names without having to write them all down using glob
    # only using dat to be what we use to pull the name. can change this is we want

    fileDirectories = glob.glob("/data/galadriel/Sricharan/Long_Integrations/{}*".format(dat))
    #print(fileDirectories)


    #print(fileDirectories)
    sTemp    = 'TemporaryDirectory_time_{}'.format((ST))
    iRes     = 'In_Resonance'
    nRes     = 'Not_In_Resonance'
    sInt     = 'Short_Integrations'


    #the following code should be set up on first use to locate and store your simulations

    destinPath = '/data/galadriel/Sricharan/Long_Integrations/'

    subDirectoriesTemp = glob.glob( '{}{}*/{}/{}'.format(destinPath,dat,sInt,sTemp)) #now have list of directory names
    #deleted the other stuff. Trying to make it things less manual, didn't need the other directories for now

    #going to start the for loop here instead instead of creating these different arrays, will do it in the loop
    # actually going a step further and just putting the plot_data.txt stuff into glob as well...
    res_plot_dat = glob.glob( '{}{}*/{}/{}/res_plot_data.txt'.format(destinPath,dat,sInt,sTemp))
    nonres_plot_dat = glob.glob( '{}{}*/{}/{}/nonres_plot_data.txt'.format(destinPath,dat,sInt,sTemp))

    plottingArr = []
    nplottingArr = []

    for j in range(3):
        plottingArr.append([])
        nplottingArr.append([])

    #going to loop through the list of different data files

    #Plotting array with resonant particles split into 3 sublists each corresponding to an orbital parameter necessary for plotting
    #semi-major axis values
    for resDat in res_plot_dat:
        #print(resDat)
        content = np.genfromtxt('{}'.format(resDat))
        #print(content)
        con0 = np.array([content[0]])
        con1 = np.array([content[1]])
        con2 = np.array([content[2]])
        #print(con0)
    
        for a in con0:
            #print("a", a)
            plottingArr[0].append(a) # semi major axis
        for e in con1:
            #print("e", e)
            plottingArr[1].append(e) # eccentricity
        for i in con2:
            plottingArr[2].append(i) # inclination
    
    """
    print("plottinArr before flatten", plottingArr[0])
    
    
    print(len(plottingArr))
    print(len(plottingArr[0]))
    # had to do this because was getting error for the case where there was only 1 resonant particle
    # error was happening because can't look through an array of 1 value
    # concatenate make the list of arrays into one array
    plottingArr[0] = np.concatenate(plottingArr[0])
    plottingArr[1] = np.concatenate(plottingArr[1])
    plottingArr[2] = np.concatenate(plottingArr[2])

    #print("plottingArr0", plottingArr[0])

    #Plotting array with nonresonant particles split into 3 sublists each corresponding to an orbital parameter necessary for plotting
    #semi-major axis values

    for nonresDat in nonres_plot_dat:
        #print(nonresDat)
        ncontent = np.genfromtxt('{}'.format(nonresDat))
        ncon0 = np.array([ncontent[0]])
        ncon1 = np.array([ncontent[1]])
        ncon2 = np.array([ncontent[2]])

        for a, e, i in zip(ncon0, ncon1, ncon2):
            #print("a", a)
            #print("e", e)
            nplottingArr[0].append(a) # semi major axis
            nplottingArr[1].append(e) # eccentricity
            nplottingArr[2].append(i) # inclination

    #print("n plotting before flatten", nplottingArr[0])

    nplottingArr[0] = np.concatenate(nplottingArr[0])
    nplottingArr[1] = np.concatenate(nplottingArr[1])
    nplottingArr[2] = np.concatenate(nplottingArr[2])

    #print("n plotting after flatten", nplottingArr[0])

    #print((len(plottingArr[0]), len(plottingArr[1])))


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


    x19 = np.linspace(19,60,18)
    x24 = np.linspace(24,60,18)
    x29 = np.linspace(29,60,18)
    x34 = np.linspace(34,60,18)
    x37 = np.linspace(37,60,18)
    #print(x)
    eccentricity_from_peri_19 = []
    eccentricity_from_peri_24 = []
    eccentricity_from_peri_29 = []
    eccentricity_from_peri_34 = []
    eccentricity_from_peri_37 = []
    for ep in range(len(x19)):
        ecc19 = 1-(19/x19[ep])
        ecc24 = 1-(24/x24[ep])
        ecc29 = 1-(29/x29[ep])
        ecc34 = 1-(34/x34[ep])
        ecc37 = 1-(37/x37[ep])
        eccentricity_from_peri_19.append(ecc19)
        eccentricity_from_peri_24.append(ecc24)
        eccentricity_from_peri_29.append(ecc29)
        eccentricity_from_peri_34.append(ecc34)
        eccentricity_from_peri_37.append(ecc37)

    
    #eccentricity vs semi major axis t = 1e7
    fig = plt.figure(figsize=(10,7))
    ax1 = plt.subplot(111)
    plt.xlim([37,42])
    plt.ylim([0, 1.1])
    plt.title('a vs. e, t = {:e}'.format(ST), fontsize = 24)
    plt.xlabel('semi major axis (AU)', fontsize = 18)
    plt.ylabel('eccentricity', fontsize = 18)
    ax1.scatter(nplottingArr[0],nplottingArr[1],marker = '.', c='r', label = 'nonresonant particles')
    ax1.scatter(plottingArr[0],plottingArr[1],marker = '.', c='b', label = 'resonant particles')

    plt.plot(x19,eccentricity_from_peri_19, color = 'darkorange', linestyle = 'dashed', label = "p = 19AU")
    plt.plot(x24,eccentricity_from_peri_24, color = 'm', linestyle = 'dashed', label = "p=23AU")
    plt.plot(x29,eccentricity_from_peri_29, color = 'forestgreen', linestyle = 'dashed', label = "p=29AU")
    plt.plot(x34,eccentricity_from_peri_34, color = 'grey', linestyle = 'dashed', label = "p=34AU")
    plt.plot(x37,eccentricity_from_peri_37, color = 'blueviolet', linestyle = 'dashed', label = "p=37AU")

    plt.axvline(x=minA, color = 'black', linestyle = '-')
    plt.axvline(x=maxA, color = 'black', linestyle = '-')
    plt.axhline(y=0.6, color = 'black', linestyle = '-')                       #some particles go slightly past 0.6      
    plt.legend(loc = 'upper right', prop={'size': 10})
    plt.show()
    plt.savefig('/data/galadriel/Sricharan/Plotting_files/{}/ave_t={}.png'.format(dat, ST))
    plt.clf()



    #inclination vs semi major axis t=0
    fig = plt.figure(figsize=(10,7))
    ax1 = plt.subplot(111)
    plt.xlim([37,42])
    #plt.ylim([0, 1.1])
    plt.ylim(0, np.pi/2)
    plt.title('a vs. i, t = {:e}'.format(ST), fontsize = 24)
    plt.xlabel('semi major axis (AU)', fontsize = 18)
    plt.ylabel('inclination', fontsize = 18)
    ax1.scatter(nplottingArr[0],nplottingArr[2],marker = '.', c='r', label = 'nonresonant particles')
    ax1.scatter(plottingArr[0],plottingArr[2],marker = '.', c='b', label = 'resonant particles')
    plt.legend(loc = 'upper right', prop={'size': 10})

    plt.axvline(x=minA, color = 'black', linestyle = '-')
    plt.axvline(x=maxA, color = 'black', linestyle = '-')
    plt.show()
    plt.savefig('/data/galadriel/Sricharan/Plotting_files/{}/avi_t={}.png'.format(dat, ST))
    plt.clf()   
    
    return ST
"""    
datDate = 'Aug302020.21.31'

create_plots(datDate, 0.0)



#create_plots(datDate, 1e7)

#create_plots(datDate, 1e8)





