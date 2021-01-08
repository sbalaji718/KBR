import rebound
import numpy as np
import random
import time 
import os as osp       
import shutil
import matplotlib.pyplot as plt
from rebound import hash as h
from ctypes import c_uint32
import csv


def moveAndCreateDir(sPath, dstDir):
    if not osp.path.isdir(dstDir):
        osp.makedirs(dstDir)
        shutil.move(sPath,dstDir)
        #print("created directory {} and moved it to {}".format(sPath, dstDir))
    else:
        # print("directory already exists")
        pass

def prob(i):
    sigma_i = 14.0*np.pi/180.  # average 15 degrees
    prob_i = np.sin(i)*np.exp(-i**2/(2.0*sigma_i**2))
    return(prob_i)

def get_i():
    keep = False
    
    while keep == False:
        num = np.random.uniform()*np.pi  # pick an inclination
        
        if np.random.uniform() < prob(num):  # decide whether to keep the inclination
            keep = True

    return(num)


def setupPlanetBinary():
    sim = rebound.Simulation()
    sim.units = ('yr','AU','Msun')

    #this is a file with m, r, x, y, z, vx, vy, vz for sun and giant planets
    with open('planetParamsCartesianAU_yr.txt', 'r') as file:
        data = file.read()

    sim.add_particles_ascii(data)

    print("---added Sun, Jupiter, Saturn, Uranus, Neptune---")

    sun=sim.particles[0]
    jupiter=sim.particles[1]
    saturn=sim.particles[2]
    uranus=sim.particles[3]
    neptune=sim.particles[4]

    sun.hash = 0
    jupiter.hash = 1
    saturn.hash = 2
    uranus.hash = 3
    neptune.hash = 4

    #simulation properties
    sim.dt = 0.2
    sim.integrator = 'whfast'
    sim.ri_whfast.safe_mode = 1
    sim.ri_whfast.corrector = 11 
    sim.move_to_com()            # move particles to center of momentum frame
    sim.N_active = 5             # number of active massive particles, includes Sun

    sim.simulationarchive_snapshot("planet-initial-conditions.bin")
    return sim


def setupSimulationFromArchive(int_count, minA, maxA, minE, maxE, Nparticles):

    sim = rebound.Simulation("planet-initial-conditions.bin")    
    print("reading in Sun and giant planets")
   
    np.random.seed(int_count)

    com = sim.calculate_com()
    #even q range
    for p in range(Nparticles):
        sem = np.random.uniform(minA,maxA)
        ecc = 1-(np.random.uniform(15.54, sem))/sem 
        sim.add(primary = com, a=sem, e=ecc, inc= get_i(), omega=np.random.uniform(0,2*np.pi),Omega=np.random.uniform(0, 2*np.pi),M=np.random.uniform(0,2*np.pi), hash = p+5) 


    print("added {} test particles".format(Nparticles))
    

    #simulation properties
    sim.dt = 0.2
    sim.integrator = 'whfast'
    sim.ri_whfast.safe_mode = 1
    #sim.ri_whfast.corrector = 11 
    sim.move_to_com()            # move particles to center of momentum frame
    sim.N_active = 5             # number of active massive particles, includes Sun

    
    return sim


def long_integration(int_count, minA, maxA, minE, maxE, integration_times, Nparticles, long_int_file= False):

    """
        running an n body integration and returning the simulation at the end
    """
    
    start_time = time.time()
    tm = time.gmtime()
    dat = time.strftime("%b%d%Y.%H.%M", tm)        

  
    sim_length = integration_times[-1]
    print("integration number {}; integrating for {} years".format(int_count, sim_length))

    #the following code should be set up on first use to locate and store your simulations
    sourcePath = '/Users/arceliahermosillo/Research/KBR/code/' 
    destinPath = '/Users/arceliahermosillo/Research/KBR/Long_Integrations/'


    if not long_int_file:
        filename = '{}_part{}_time{}_A_{:.3f}-{:.3f}_iSig_{}_E_{:.3f}-{:.3f}_even_q_{}'.format(dat,
            Nparticles, sim_length, minA, maxA, 14, minE, maxE, int_count)
    else:
        filename = long_int_file

    # check to see if long int already exists. If so, nothing else needs to happen. If it doesn't then we continue with long integration

    # ------- Manual Snapshots --------------------------
    try:
        sim = rebound.Simulation("{}{}.bin".format(destinPath,filename))
        print("tried loading simulation archive\n")
    except:
        print("failed to load simulation archive, loading initial conditions")
        sim = setupSimulationFromArchive(int_count, minA, maxA, minE, maxE, Nparticles)
        for t in integration_times:
            sim.integrate(t, exact_finish_time=0)
            print("done: {} ; N particles: {}".format(t, sim.N))
            print("time passed: {}".format(time.time() - start_time))
            sim.simulationarchive_snapshot('{}.bin'.format(filename))

        print("long integration is done")
    
    print("long Integration took {}".format(time.time() - start_time))
    sourceFile = '{}{}.bin'.format(sourcePath,filename)
    destDir    = '{}{}'.format(destinPath,filename)
    moveAndCreateDir(sourceFile,destDir)

    return sim, filename 


def make_shortint_directories(destinPath, filename, ST, dat):
    #the following code should be set up on first use to locate and store your simulations
    sInt     = 'Short_Integrations_{}'.format(dat)
    sTemp    = 'Short_Integration_time_{}'.format(np.round(ST))
    iRes     = 'In_Resonance'
    nRes     = 'Not_In_Resonance'

    global subDirTemp 
    global irDir      
    global nrDir 

    mainDir    = '{}{}/'.format(destinPath,filename)
    dirName    = '{}{}/{}'.format(destinPath,filename,sInt)    
    subDirTemp = '{}{}/{}/{}'.format(destinPath,filename,sInt,sTemp)
    irDir      = '{}{}/{}/{}/{}'.format(destinPath,filename,sInt,sTemp,iRes)
    nrDir      = '{}{}/{}/{}/{}'.format(destinPath,filename,sInt,sTemp,nRes) 

    try:
        osp.mkdir(destinPath)
        #print('Directory',destinPath,'created.')
    except FileExistsError:
        pass
        #print('Directory',destinPath,'already exists.')

    try:
        osp.mkdir(dirName)
        #print ('Directory',dirName,'created.')
    except FileExistsError:
        pass
        #print("Directory",dirName,"already exists.")

    try:
        osp.mkdir(subDirTemp)
        #print ('Directory',subDirTemp,'created.')
    except FileExistsError:
        pass
        #print("Directory",subDirTemp,"already exists.")

    try:
        osp.mkdir(irDir)
        #print ('Directory',irDir,'created.')
    except FileExistsError:
        pass
        #print("Directory",irDir,"already exists.")

    try:
        osp.mkdir(nrDir)
        #print ('Directory',nrDir,'created.')
    except FileExistsError:
        pass
        #print("Directory",nrDir,"already exists.")
    return True


def short_integration(int_count, simarchive, sim_length, indexSimulation, filename):

    start_time = time.time()

    tm = time.gmtime()
    dat= time.strftime("%b%d", tm)
    
    destinPath = '/Users/arceliahermosillo/Research/KBR/Long_Integrations/'
    longInt    = '{}{}/{}'.format(destinPath,filename,filename)

    sa = rebound.SimulationArchive("{}.bin".format(longInt))
    sim = sa[indexSimulation] ## see comment above for this 
    ST = sim.t             #(the snapshot time we're using as initial conditions)
   
    make_shortint_directories(destinPath, filename, ST, dat)

    IT = sim_length         #this is the short integration run time.
    ET = ST + IT
    Nout = 1000
    Nparticles = sim.N -1 #number of objects without the sun
    npart = sim.N - sim.N_active # number of test particles
    # sim.exit_max_distance = maxDistance

    print("starting short integration {} with start time {}".format(int_count, ST))
    start_sim_time = time.time()
    short_filename = "{}_short+1e5.bin".format(filename)
    sim.automateSimulationArchive("{}/{}".format(subDirTemp,short_filename), IT/Nout)
    sim.integrate(ET, exact_finish_time = 0)

    print("short integration took {} seconds".format(time.time() - start_sim_time))
    print(short_filename)
    return short_filename  

    # then make a function that checks for resonance and makes a few plots

def check_resonance_make_plots(short_filename):
    print(short_filename)
    short_bin = rebound.SimulationArchive("{}/{}".format(subDirTemp,short_filename))

    print("short_bin")
    print(short_bin)

    print("short_bin[-1]")
    print(short_bin[-1])

    Nparticles = short_bin[-1].N
    Nout = len(short_bin)
    ST = short_bin.tmax

    print("Nparticles: {}".format(Nparticles))
    print("Nout: {}".format(Nout))
    print("ST: {}".format(ST))


    ################### ------------- arrays to record values ------------ #####################

    ax  = np.empty((Nparticles,Nout))
    ecc = np.empty((Nparticles,Nout))
    inc = np.empty((Nparticles,Nout))
    lam = np.empty((Nparticles,Nout))
    pom = np.empty((Nparticles,Nout))
    phi = np.zeros((Nparticles,Nout))

    lasc_node = np.empty((Nparticles,Nout))
    arg_peri = np.empty((Nparticles,Nout))
    t_anom = np.empty((Nparticles,Nout))
    M_anom = np.empty((Nparticles,Nout))
    peri = np.empty((Nparticles,Nout))
    xvals = np.empty((Nparticles,Nout))
    yvals = np.empty((Nparticles,Nout))

    deltaTheta = np.empty((Nparticles, Nout)) # needed to make rotation to observed Neptune frame


    time = np.empty(Nout)

    ######################--------------- record values into arrays ------####################
    ct = 0
    for i, sim in enumerate(short_bin):
        ct += 1
        n = sim.N
        # print(len(short_bin))
        hashesParticles = np.zeros(sim.N,dtype="uint32")
        sim.serialize_particle_data(hash=hashesParticles)
        time[i] = sim.t
        com = sim.calculate_com()
        for j in range(n-1):
            p = sim.particles[j+1]
            o = p.calculate_orbit(com)
            # print("hash: {}".format(p.hash.value))
            ax[p.hash.value][i] = o.a
            ecc[p.hash.value][i] = o.e
            inc[p.hash.value][i] = o.inc
            lam[p.hash.value][i] = o.l
            pom[p.hash.value][i] = o.pomega
            lasc_node[p.hash.value][i] = o.Omega
            arg_peri[p.hash.value][i] = o.omega
            t_anom[p.hash.value][i] = o.f 
            M_anom[p.hash.value][i] = o.M 
            peri[p.hash.value][i] = ax[p.hash.value][i]*(1- ecc[p.hash.value][i])
            xvals[p.hash.value][i] = p.x 
            yvals[p.hash.value][i] = p.y

    
    # calculate phi and deltaTheta here since need fully recorded arrays
    for i, sim in enumerate(short_bin):
        n = sim.N
        for j in range(n):

            phi[j][i] = (3*lam[j][i] - 2*lam[4][i] - pom[j][i])%(2*np.pi)

            Neptune_xsim = xvals[4][i]
            Neptune_ysim = yvals[4][i]

            xsim = xvals[j][i]
            ysim = yvals[j][i]
            theta_sim = np.arctan2(ysim,xsim)

            xsur = 26.85758046958696
            ysur = -13.32890006819031
            # for julian date ? FILL THIS OUT
            theta_sur = np.arctan2(ysur,xsur)

            #difference of both angles
            new_theta = theta_sim - theta_sur
            deltaTheta[j][i] = new_theta


    #Applying rotation to array of lasc values by adding rotate_diff values to corresponding values in rotated_longitude array
    rotated_longitude = np.empty((Nparticles, Nout))

    for i in range(len(lasc_node[0])):
        for j in range(len(lasc_node)):
            rotated_longitude[j][i] = lasc_node[j][i] - deltaTheta[3][i]

    phiAmp = []

    for i in range(len(phi)):
        pamp = (np.max(phi[i]) - np.min(phi[i]))/2
        phiAmp.append(pamp)

    #################### ------------ NOW check for resonance -----------  ################

    resonant_particles = []
    nonresonant_particles = [] 
    count = 0   
    count_n = 0     

    phi_min = 5*np.pi/180
    phi_max = 355*np.pi/180

    for i in range(Nparticles):
        try:
            if (all(phi[i] < phi_max) and all(phi[i] > phi_min)):

                print("in resonance")
                print(i)
                resonant_particles.append(i)
                count +=1

                plt.figure(figsize=(15,10))
                plt.title('Resonant angle libration', fontsize = 24)
                plt.xlabel('Time(years)', fontsize = 18)
                plt.ylabel('Resonant argument (degrees)', fontsize = 18)
                plt.scatter(time,phi[i], marker = '.',s = 10)
                plt.ylim(0, 2*np.pi)
                plt.savefig('{}/Particle {} Phi vs Time Plot.png'.format(irDir,i))  
                plt.clf()

            else: 
                nonresonant_particles.append(i)
                print("not in resonance")
                print(i)
                count_n +=1

                plt.figure(figsize=(15,10))
                plt.title('Resonant angle circulation', fontsize = 24)
                plt.xlabel('Time(years)', fontsize = 18)
                plt.ylabel('Resonant argument (degrees)', fontsize = 18)
                plt.scatter(time,phi[j], marker = '.',s = 10)
                plt.ylim(0,2*np.pi)
                plt.savefig('{}/Particle {} Phi vs Time Plot.png'.format(nrDir,j))  
                plt.clf()
        except RuntimeWarning:
            print(phi[i])


    with open("{}/Particles_in_resonance_{}.txt".format(subDirTemp, np.round(ST)), "w+") as my_file:                               
        for i in resonant_particles:
             my_file.write(str(i)+"\n")

    with open("{}/Particles_not_in_resonance_{}.txt".format(subDirTemp, np.round(ST)), "w+") as my_file:                               
        for j in nonresonant_particles:
            my_file.write(str(j)+"\n")

    print("{} particles in resonance".format(count))
    print("{} particles not in resonance".format(count_n))


    ######### --------- now make file with all values ---------- ########

    data_arr = []

    for particle in range(len(lam)):
        data_arr.append([])   
        for integration in range(len(lam[0])):
            #print("Integration: " + str(integration) + "Num: " + str(num))
            data_arr[particle].append(hashesParticles[particle])
            data_arr[particle].append(peri[particle][integration])
            data_arr[particle].append(ax[particle][integration]) 
            data_arr[particle].append(ecc[particle][integration])
            data_arr[particle].append(inc[particle][integration])
            data_arr[particle].append(lasc_node[particle][integration])
            data_arr[particle].append(arg_peri[particle][integration])
            data_arr[particle].append(M_anom[particle][integration])
            data_arr[particle].append(t_anom[particle][integration])
            data_arr[particle].append(phi[particle][integration])
            data_arr[particle].append(rotated_longitude[particle][integration])
            data_arr[particle].append(phiAmp[particle])
            data_arr[particle].append(xvals[particle][integration])
            data_arr[particle].append(yvals[particle][integration])
            if particle in resonant_particles:
                data_arr[particle].append(True)
            else:
                data_arr[particle].append(False)   

    data_arr = np.array(data_arr).reshape(len(lam)*len(lam[0]), 15) # (numParticles*numTimesteps, numOutputs)     

    with open('{}/{}_data_array_{}.csv'.format(subDirTemp,short_filename, ST), mode = 'w') as file:
       datawriter = csv.writer(file, delimiter = ',')
       datawriter.writerow(['pnumber', 'peri', 'a', 'e', 'i', 'Omega', 'w', 'f', 'M', 'phi', 'Omega_rot', 'libAmp', 'x', 'y', 'resonance'])
       for d in data_arr:
           datawriter.writerow(d)

    return 0

    """
    #data array
    data_arr = []

    for particle in range(len(l)):
        data_arr.append([])   
        for integration in range(len(l[0])):
            #print("Integration: " + str(integration) + "Num: " + str(num))
            data_arr[particle].append(hashesParticles[particle])
            data_arr[particle].append(peri[particle][integration])
            data_arr[particle].append(a[particle][integration]) 
            data_arr[particle].append(e[particle][integration])
            data_arr[particle].append(incl[particle][integration])
            data_arr[particle].append(lasc_node[particle][integration])
            data_arr[particle].append(arg_peri[particle][integration])
            data_arr[particle].append(M_anom[particle][integration])
            data_arr[particle].append(t_anom[particle][integration])
            data_arr[particle].append(phi[particle][integration])
            data_arr[particle].append(rotated_longitude[particle][integration])
            data_arr[particle].append(phiAmp[particle])
            data_arr[particle].append(xvals[particle][integration])
            data_arr[particle].append(yvals[particle][integration])
            if particle in resonant_particles:
                data_arr[particle].append(True)
            else:
                data_arr[particle].append(False)   

    data_arr = np.array(data_arr).reshape(len(l)*len(l[0]), 15) # (numParticles*numTimesteps, numOutputs)     

    #with open('{}/{}_data_array_{}.csv'.format(subDirTemp,filename, ST), mode = 'w') as file:
    #    datawriter = csv.writer(file, delimiter = ',')
    #    datawriter.writerow(['pnumber', 'peri', 'a', 'e', 'i', 'Omega', 'w', 'f', 'M', 'phi', 'Omega_rot', 'libAmp', 'x', 'y', 'resonance'])
     #   for d in data_arr:
     #       datawriter.writerow(d)





    #Writing a,e, and i values to read in Plotting script

    res_plot_data = []
    nonres_plot_data = []    

    ar = []
    er = []
    ir = []
    for i in a[resonant_particles]:
        ar.append(i[0])
    for j in e[resonant_particles]:
        er.append(j[0])
    for k in incl[resonant_particles]:
        ir.append(k[0])  
    res_plot_data.append(ar) 
    res_plot_data.append(er) 
    res_plot_data.append(ir)  
    np.savetxt('{}/res_plot_data.txt'.format(subDirTemp), res_plot_data, fmt = '%s')

    anr = []
    enr = []
    inr = []
    for i in a[nonresonant_particles]:
        anr.append(i[0])
    for j in e[nonresonant_particles]:
        enr.append(j[0])
    for k in incl[nonresonant_particles]:
        inr.append(k[0])
    nonres_plot_data.append(anr) 
    nonres_plot_data.append(enr) 
    nonres_plot_data.append(inr)  
    np.savetxt('{}/nonres_plot_data.txt'.format(subDirTemp), nonres_plot_data, fmt = '%s')

    print('DONE!')
    print("short integration {} took {}".format(integrationN, time.time() - start_simul_time))

    return sim
"""

