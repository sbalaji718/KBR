import rebound
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import csv
import pdb
import glob
import time    
import os as osp       
import shutil
import itertools
from rebound import hash as h
from ctypes import c_uint32
import pandas as pd



def moveAndCreateDir(sPath, dstDir):
    if not osp.path.isdir(dstDir):
        osp.makedirs(dstDir)
        shutil.move(sPath,dstDir)
        #print("created directory {} and moved it to {}".format(sPath, dstDir))
    else:
    	# print("directory already exists")
    	pass

def prob(i):
    sigma_i = 12.0*np.pi/180.  # average 15 degrees
    prob_i = np.sin(i)*np.exp(-i**2/(2.0*sigma_i**2))
    return(prob_i)

def get_i():
    keep = False
    
    while keep == False:
        num = np.random.uniform()*np.pi  # pick an inclination
        
        if np.random.uniform() < prob(num):  # decide whether to keep the inclination
            keep = True

    return(num)

def long_integration(i, minA, maxA, minE, maxE, imax, totalExpTime, Nparticles):
    

    """runs an nbody integrationfor the giant planets + Nparticles test particles. 
    It goes for totalTime year
    it returns minA, maxA, minE, maxE and creates a bin file with the simulation archive
    """
    
    start_time = time.time()
    tm = time.gmtime()
    dat= time.strftime("%b%d%Y.%H.%M", tm)        

    base = 30*(3./2.)**(2./3.) #~39AU

    minQ = (minA*(1.-maxE))            #Perihelion distance
    maxQ = (maxA*(1.+maxE))            #Apehelion distance
    #maxI =  maxI*(np.pi/180)           #maximum inclination argument
    Nout = 1000 

    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    print("starting integration {}".format(i))

    totalTime = 10**totalExpTime

    print("set units as yr, AU, Msun")
    print("Integration is running for {} years for giant planets plus {} test particles".format(totalTime, Nparticles))

    #this is a file with m, r, x, y, z, vx, vy, vz for sun and giant planets
    # they are barycenter values from NASA Horizons
    with open('planetParamsCartesianAU_yr.txt', 'r') as file:
        data = file.read()
        
    sim.add_particles_ascii(data)

    print("---added sun, jupiter, saturn, uranus, neptune---")


    sun=sim.particles[0]
    jupiter=sim.particles[1]
    saturn=sim.particles[2]
    uranus=sim.particles[3]
    neptune=sim.particles[4]

    sun.hash = 1
    jupiter.hash = 2
    saturn.hash = 3
    uranus.hash = 4
    neptune.hash = 5

    np.random.seed(i)

    """
    #even e range
    for p in range(Nparticles):
        sim.add(a=np.random.uniform(minA, maxA), e=np.random.uniform(minE,maxE), inc= get_i(), omega=np.random.uniform(0,2*np.pi),Omega=np.random.uniform(0, 2*np.pi),M=np.random.uniform(0,2*np.pi), hash = p+6)
    """

    #even q range
    for p in range(Nparticles):
        sem = np.random.uniform(minA,maxA)
        ecc = 1-(np.random.uniform(15.5, sem))/sem 
        sim.add(a=sem, e=ecc, inc= get_i(), omega=np.random.uniform(0,2*np.pi),Omega=np.random.uniform(0, 2*np.pi),M=np.random.uniform(0,2*np.pi), hash = p+6) 

    sim.dt = 0.2
    sim.integrator = 'whfast'
    #sim.ri_whfast.safe_mode = 0
    #sim.ri_whfast.corrector = 11 
    sim.move_to_com()            # move particles to center of momentum frame
    sim.N_active = 5             # number of active massive particles, includes Sun
    #sim.exit_max_distance = maxDistance
    #sim.status()

    #the following code should be set up on first use to locate and store your simulations
    master_dir = '{}_part{}_time{}_A_{:.3f}-{:.3f}_Q_{:.3f}-{:.3f}_I_{:.3f}_E_{:.3f}-{:.3f}_even_q'.format(dat,Nparticles,totalTime,minA,maxA,minQ,maxQ,imax,minE,maxE)
    

    filename = '{}_part{}_time{}_A_{:.3f}-{:.3f}_Q_{:.3f}-{:.3f}_I_{:.3f}_E_{:.3f}-{:.3f}_even_q_{}'.format(dat,Nparticles,totalTime,minA,maxA,minQ,maxQ,imax,minE,maxE,i)
    
    
    sourcePath = '/data/galadriel/Sricharan/KBO/KBR/code/' 
    destinPath = '/data/galadriel/Sricharan/KBO/KBR/Long_Integrations/{}/'.format(master_dir)
    #print(destinPath)
    
    
    
    #----------------verification that sim is running------------------
    #def heartbeat(sim):         
    #   print(sim.contents.t)
    #sim.heartbeat = heartbeat
    
    
    #--------Automated Simulation Archive--------
    #sim.automateSimulationArchive('{}.bin'.format(filename), interval = intervalTime, deletefile = True)
    #sim.integrate(totalTime, exact_finish_time = 0)

    # ------- Manual Snapshots --------------------------

    realtimes = np.logspace(1, totalExpTime, totalExpTime)
    realtimes = np.insert(realtimes, 0, 0)

    EscParticles = []
    for i,t in enumerate(realtimes):
        while sim.t < t:
            #try:
                #print('Integrating for {} years'.format(t))
            sim.integrate(t, exact_finish_time=0)
                #print(sim.t)
            #except rebound.Escape as error:
                #print('A particle escaped at {} years'.format(sim.t))
                #for j in range(sim.N):
                    #p = sim.particles[j]
                    #dist = p.x*p.x + p.y*p.y + p.z*p.z
                    #if dist > sim.exit_max_distance**2:
                        #index = p.hash
                #sim.remove(hash=index)
                #EscParticles.append(index)
        print("done: {}".format(t))
        print("time passed: {}".format(time.time() - start_time))
        sim.simulationarchive_snapshot('{}.bin'.format(filename))

    print("long integration is done")
   
    try: 
        osp.mkdir(destinPath)
        print('Directory',destinPath,'created.')
    except FileExistsError: 
        print('Directory',destinPath,'already exists.')    


    sourceFile = '{}{}.bin'.format(sourcePath,filename)
    destDir    = '{}{}'.format(destinPath,filename)
    moveAndCreateDir(sourceFile,destDir)

    data = np.transpose([minA, maxE, minE, maxE, Nparticles, totalTime, filename])
    
    #--------this cell will pull and output the orbital elements into a .txt--------------

    txt = open("{}/Orbital_Elements.txt".format(destDir),"w")

    txt.write("NUMBER OF PARTICLES      {:3d}\r\n".format(Nparticles))
    txt.write("SEMI-MAJOR AXIS        (A) {:.3f}-{:.3f}\r\n".format(minA,maxA))
    txt.write("PERIHELION RANGE       (Q) {:.3f}-{:.3f}\r\n".format(minQ,maxQ))
    txt.write("ECCENTRICITY RANGE     (E) {:.3f}-{:.3f}\r\n".format(minE,maxE))
    txt.write("NUMBER OF TIMESTEPS    {:3d}\r\n".format(Nout))

    txt.close()
        
    
    print("long Integration took {}".format(time.time() - start_time))
        
     
    return minA, maxA, minE, maxE, imax, Nparticles, totalTime, filename



def short_integration(dat, Nparticles, totalExpTime, integrationN, minA, maxA, minE, maxE, imax, shortTime, fileName, indexSimulation):

    start_simul_time = time.time()
    #tm = time.gmtime()
    #dat= time.strftime("%b%d%Y.%H.%M", tm) 
    
    minQ = (minA*(1.-maxE))           
    maxQ = (maxA*(1.+maxE))
    totalTime = 10**totalExpTime
    
    
    #the following code should be set up on first use to locate and store your simulations
    master_dir = '{}_part{}_time{}_A_{:.3f}-{:.3f}_Q_{:.3f}-{:.3f}_I_{:.3f}_E_{:.3f}-{:.3f}_even_q'.format(dat,Nparticles,totalTime,minA,maxA,minQ,maxQ,imax,minE,maxE)
    
    destinPath = '/data/galadriel/Sricharan/KBO/KBR/Long_Integrations/{}/'.format(master_dir)
    #destinPath = '/data/galadriel/Sricharan/Long_Integrations/'

    longInt    = '{}{}/{}'.format(destinPath,fileName,fileName)
    #print(longInt)
    sa = rebound.SimulationArchive("{}.bin".format(longInt)) #names archive object 
    sim = sa[indexSimulation] ## see comment above for this 
    ST = sim.t             #(the snapshot time we're using as initial conditions)
    print("simulation time: {}".format(ST))
    print("particles N: {}".format(sim.N))
    # doing this just for naming purposes


    #orbits = sim.calculate_orbits()
    #for orbit in orbits:
        #print(orbit)

        
    sTemp    = 'TemporaryDirectory_time_{}'.format(np.round(ST))
    iRes     = 'In_Resonance'
    nRes     = 'Not_In_Resonance'
    sInt     = 'Short_Integrations'
    
    

    mainDir    = '{}{}/'.format(destinPath,fileName)
    dirName    = '{}{}/{}'.format(destinPath,fileName,sInt)
    subDirTemp = '{}{}/{}/{}'.format(destinPath,fileName,sInt,sTemp)
    irDir      = '{}{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,iRes)
    nrDir      = '{}{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,nRes)

    try: 
        osp.mkdir(destinPath)
        print('Directory',destinPath,'created.')
    except FileExistsError: 
        print('Directory',destinPath,'already exists.')    

    try:
        osp.mkdir(dirName)
        print ('Directory',dirName,'created.')
    except FileExistsError:
        print("Directory",dirName,"already exists.")

    try:
        osp.mkdir(subDirTemp)
        print ('Directory',subDirTemp,'created.')
    except FileExistsError:
        print("Directory",subDirTemp,"already exists.")

    try:
        osp.mkdir(irDir)
        print ('Directory',irDir,'created.')
    except FileExistsError:
        print("Directory",irDir,"already exists.")

    try:
        osp.mkdir(nrDir)
        print ('Directory',nrDir,'created.')
    except FileExistsError:
        print("Directory",nrDir,"already exists.")


    IT = np.round(shortTime)          #this is the short integration run time.
    ET = ST + IT
    Ntotal = sim.N #number of objects without the sun
    npart = sim.N - sim.N_active # number of test particles
    Nout = 1000

    print("starting short integration {} with start time {}".format(integrationN, ST))
    
    # pointer to get the hashes of particles still alive

    hashesParticles = np.zeros(sim.N,dtype="uint32")
    sim.serialize_particle_data(hash=hashesParticles)
    #print(hashesParticles)

    print("now running short integration from {} with {} particles + giant planets + sun".format(sim.t, npart))

    intTimes = np.linspace(ST, ET, Nout)
    #print(intTimes)

    ### ------------ where we are going to store values ----------------- #####
    # these are 2D matrices that store the values for each particle, for each timestep
    # made them have 9999 so it's noticeable where particles escaped (values didn't store)

    l = np.zeros((Ntotal,Nout))
    p = np.zeros((Ntotal,Nout))
    a = np.ones((Ntotal,Nout))*9999
    e = np.ones((Ntotal,Nout))*9999
    lasc_node = np.ones((Ntotal,Nout))*9999
    arg_peri = np.zeros((Ntotal,Nout))
    t_anom = np.ones((Ntotal,Nout))*9999
    incl = np.ones((Ntotal,Nout))*9999
    phi = np.zeros((Ntotal,Nout))
    M_anom = np.ones((Ntotal,Nout))*9999
    xvals = np.ones((Ntotal,Nout))*9999
    yvals = np.ones((Ntotal,Nout))*9999
    zvals = np.ones((Ntotal,Nout))*9999
    timeArray = np.ones(Nout)*9999

    mln_arr = []
    mlp_arr = []
    pj_arr = []
    a_arr = []
    e_arr = []
    inc_arr = []
    lasc_node_arr = []
    arg_peri_arr = []
    t_anom_arr = []
    M_anom_arr = []
    x_arr = []
    y_arr = []
    z_arr = []

    #print(sim.exit_max_distance)

    for i,times in enumerate(intTimes):
        while sim.t < times:
            #try:
            sim.integrate(times, exact_finish_time = 0)
            #except rebound.Escape as error:
                #print(error)
                #for part in range(sim.N):
                    #psim = sim.particles[part]
                    #dist = psim.x*psim.x + psim.y*psim.y + psim.z*psim.z
                    #if dist > sim.exit_max_distance**2:
                        #index = psim.hash
    #                     print("particle with hash {} escaped".format(index))
                #sim.remove(hash=index)
        for j, j_hash in enumerate(hashesParticles[1:]):
    #         print("j: {} ; hash: {}".format(j, j_hash))
            #try: 
            ps = sim.particles[h(c_uint32(j_hash))]
            l[j][i] = ps.calculate_orbit().l
            p[j][i] = ps.calculate_orbit().pomega
            a[j][i] = ps.calculate_orbit().a
            e[j][i] = ps.calculate_orbit().e
            incl[j][i] = ps.calculate_orbit().inc
            lasc_node[j][i] = ps.calculate_orbit().Omega 
            arg_peri[j][i] = ps.calculate_orbit().omega
            t_anom[j][i] = ps.calculate_orbit().f
            M_anom[j][i] = ps.calculate_orbit().M
            xvals[j][i] = ps.x
            yvals[j][i] = ps.y
            zvals[j][i] = ps.z
            #except rebound.ParticleNotFound as error: 
                # since particles escaping as we store/integrate
                #pass
    #             print("idk {}".format(error))

            #renaming values
            mlp = l[j][i]
            pj = p[j][i]
            mln = l[3][i]
            sem = a[j][i]
            ecc = e[j][i]
            inc = incl[j][i]
            lan = lasc_node[j][i]
            ap = arg_peri[j][i]
            ta = t_anom[j][i]
            ma = M_anom[j][i]
            x = xvals[j][i]
            y = yvals[j][i]
            z = zvals[j][i]


            #appending to cleaned up arrays
            mln_arr.append(mln)
            mlp_arr.append(mlp)
            pj_arr.append(pj)
            a_arr.append(sem)
            e_arr.append(ecc)
            inc_arr.append(inc)
            lasc_node_arr.append(lan)
            arg_peri_arr.append(ap)
            t_anom_arr.append(ta)
            M_anom_arr.append(ma)
            x_arr.append(x)
            y_arr.append(y)
            z_arr.append(z)

            phi_temp = 3.*mlp - 2.*mln - pj   
            phi[j][i] = phi_temp%(2*np.pi)

    print("done: after short int {} particles left".format(sim.N))


    resonant_particles = []       
    for i in range(Ntotal):
        phi_i = phi[i]
        if (all(phi[i] < 355*np.pi/180) and all(phi[i] > 5*np.pi/180)):
            resonant_particles.append(i)
        with open("{}/Particles_in_resonance_{}.txt".format(subDirTemp, np.round(ST)), "w+") as my_file:                               
            for i in resonant_particles:
                 my_file.write(str(i)+"\n")



    nonresonant_particles = []       
    for j in range(Ntotal):
        phi_n = phi[j]
        if (any(phi[j] > 355*np.pi/180) and any(phi[j] < 5*np.pi/180)):
            nonresonant_particles.append(j)
        with open("{}/Particles_not_in_resonance_{}.txt".format(subDirTemp, np.round(ST)), "w+") as my_file:                               
            for i in nonresonant_particles:
                my_file.write(str(i)+"\n")



    deltaTheta = np.zeros((Ntotal, Nout))
    deltaThetaArr = []

    rotate_diff = []
    for i in range(Nout):
        for j in range(Ntotal):
            Neptune_xsim = xvals[3][i]
            Neptune_ysim = yvals[3][i]

            xsim = xvals[j][i]
            ysim = yvals[j][i]
            theta_sim = np.arctan2(ysim,xsim)


            xsur = 2.685758046958696E+01   
            ysur = -1.332890006819031E+01
            theta_sur = np.arctan2(ysur,xsur)

    
            #difference of both angles
            new_theta = theta_sim - theta_sur
            deltaTheta[j][i] = new_theta
            deltaThetaArr.append(new_theta)
            #new theta is added to the lasc of all the giant planets and the test particles

            rotate_diff.append(new_theta)

    #Applying rotation to array of lasc values by adding rotate_diff values to corresponding values in rotated_longitude array
    rotated_longitude = np.zeros((Ntotal, Nout))

    for i in range(len(lasc_node[0])):
        for j in range(len(lasc_node)):
            rotated_longitude[j][i] = lasc_node[j][i] - deltaTheta[3][i]


    #data array
    data_arr = []

    for particle in range(len(l)):
        data_arr.append([])   
        for integration in range(len(l[0])):
            data_arr[particle].append(hashesParticles[particle])
            data_arr[particle].append(a[particle][integration]) 
            data_arr[particle].append(e[particle][integration])
            data_arr[particle].append(incl[particle][integration])
            data_arr[particle].append(lasc_node[particle][integration])
            data_arr[particle].append(arg_peri[particle][integration])
            data_arr[particle].append(M_anom[particle][integration])
            data_arr[particle].append(phi[particle][integration])
            data_arr[particle].append(deltaTheta[particle][integration])
            data_arr[particle].append(xvals[particle][integration])
            data_arr[particle].append(yvals[particle][integration])
            data_arr[particle].append(zvals[particle][integration])
            if particle in resonant_particles:
                data_arr[particle].append(True)
            else:
                data_arr[particle].append(False)   

    data_arr = np.array(data_arr).reshape(len(l)*len(l[0]), 13) # (numParticles*numTimesteps, numOutputs)     

    with open('{}/{}_data_array.csv'.format(subDirTemp,fileName), mode = 'w+') as file:
        datawriter = csv.writer(file, delimiter = ',')
        datawriter.writerow(['pnumber', 'a', 'e', 'i', 'Omega', 'w', 'M_anom', 'phi', 'dTheta', 'x', 'y','z', 'resonance'])
        for d in data_arr:
            datawriter.writerow(d)



    A = []
    E = []
    I = []
    N = []
    P = []
    M = []
    F = []


    X = []
    Y = []
    Z = []
    

    for semimajor in a[resonant_particles]:
        A.append(semimajor)
    for eccentricity in (e[resonant_particles]):
        E.append(eccentricity)
    for inclination in (incl[resonant_particles]):
        I.append(inclination)
    for longRot in rotated_longitude[resonant_particles]:
        N.append(longRot) 
    for argument in (arg_peri[resonant_particles]):
        P.append(argument)    
    for mean in (M_anom[resonant_particles]):
        M.append(mean)
    for trueAn in (t_anom[resonant_particles]):
        F.append(trueAn)

    for val1 in xvals[resonant_particles]:
        X.append(val1)
    for val2 in yvals[resonant_particles]:
        Y.append(val2)
    for val3 in zvals[resonant_particles]:
        Z.append(val3)

    A_iter = itertools.chain.from_iterable(A)
    E_iter = itertools.chain.from_iterable(E)
    I_iter = itertools.chain.from_iterable(I)
    N_iter = itertools.chain.from_iterable(N)
    P_iter = itertools.chain.from_iterable(P)
    M_iter = itertools.chain.from_iterable(M)
    F_iter = itertools.chain.from_iterable(F)

    X_iter = itertools.chain.from_iterable(X)
    Y_iter = itertools.chain.from_iterable(Y)
    Z_iter = itertools.chain.from_iterable(Z)

    A1 = list(A_iter)
    E1 = list(E_iter)
    I1 = list(I_iter)
    N1 = list(N_iter)
    P1 = list(P_iter)
    M1 = list(M_iter)
    F1 = list(F_iter)

    X1 = list(X_iter)
    Y1 = list(Y_iter)
    Z1 = list(Z_iter)

    data=np.transpose([A1,E1,I1,N1,P1,M1,X1,Y1,Z1])
    np.savetxt('{}/Survey_data_{}'.format(subDirTemp, integrationN),data,fmt='%s')
    """
    cartesian_data=np.transpose([A1, E1, I1, N1, P1, F1, X1,Y1])
    np.savetxt('{}/data_for_checking_resonance_{}'.format(subDirTemp, integrationN),cartesian_data,fmt='%s')
    """

    #Writing a,e, and i values to read in Plotting script

    res_plot_data = []
    nonres_plot_data = []    

    ar = []
    er = []
    ir = []
    Or = []
    omr = []
    Mr = []
    for i in a[resonant_particles]:
        ar.append(i[0])
    for j in e[resonant_particles]:
        er.append(j[0])
    for k in incl[resonant_particles]:
        ir.append(k[0])
    for l in lasc_node[resonant_particles]:
        Or.append(l[0])
    for m in arg_peri[resonant_particles]:
        omr.append(m[0])
    for n in M_anom[resonant_particles]:
        Mr.append(n[0])    
    res_plot_data.append(ar) 
    res_plot_data.append(er) 
    res_plot_data.append(ir) 
    res_plot_data.append(Or) 
    res_plot_data.append(omr) 
    res_plot_data.append(Mr)  
    np.savetxt('{}/res_plot_data.txt'.format(subDirTemp), res_plot_data, fmt = '%s')

    anr = []
    enr = []
    inr = []
    Onr = []
    omnr = []
    Mnr = []
    for i in a[nonresonant_particles]:
        anr.append(i[0])
    for j in e[nonresonant_particles]:
        enr.append(j[0])
    for k in incl[nonresonant_particles]:
        inr.append(k[0])
    for l in lasc_node[nonresonant_particles]:
        Onr.append(l[0])
    for m in arg_peri[nonresonant_particles]:
        omnr.append(m[0])
    for n in M_anom[nonresonant_particles]:
        Mnr.append(n[0])    
    nonres_plot_data.append(anr) 
    nonres_plot_data.append(enr) 
    nonres_plot_data.append(inr)  
    nonres_plot_data.append(Onr) 
    nonres_plot_data.append(omnr) 
    nonres_plot_data.append(Mnr)
    np.savetxt('{}/nonres_plot_data.txt'.format(subDirTemp), nonres_plot_data, fmt = '%s')

    print('DONE!')
    print("short integration {} took {} seconds".format(integrationN, time.time() - start_simul_time))

    return sim
    
    
#Units specified in Long integration
#Units currently set to ('yr','AU','Msun')




def kozai_check(dat, Nparticles, totalExpTime, integrationN, minA, maxA, minE, maxE, imax, shortTime, fileName, indexSimulation):
    start_simul_time = time.time()

    
    minQ = (minA*(1.-maxE))           
    maxQ = (maxA*(1.+maxE))
    totalTime = 10**totalExpTime
    
    
    #the following code should be set up on first use to locate and store your simulations
    master_dir = '{}_part{}_time{}_A_{:.3f}-{:.3f}_Q_{:.3f}-{:.3f}_I_{:.3f}_E_{:.3f}-{:.3f}_even_q'.format(dat,Nparticles,totalTime,minA,maxA,minQ,maxQ,imax,minE,maxE)
    
    destinPath = '/data/galadriel/Sricharan/KBO/KBR/Long_Integrations/{}/'.format(master_dir)
    
    longInt    = '{}{}/{}'.format(destinPath,fileName,fileName)
    
    
    
    sa = rebound.SimulationArchive("{}.bin".format(longInt)) #names archive object 
    sim = sa[indexSimulation] ## see comment above for this 
    ST = np.round(sim.t)             #(the snapshot time we're using as initial conditions)
    
    
    #print("simulation time: {}".format(ST))
    #print("particles N: {}".format(sim.N))
    # doing this just for naming purposes
    
    #sim = rebound.Simulation()
    #sim.units = ('yr', 'AU', 'Msun')

    
        
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
        osp.mkdir(destinPath)
        print('Directory',destinPath,'created.')
    except FileExistsError: 
        print('Directory',destinPath,'already exists.')    

    try:
        osp.mkdir(dirName)
        print ('Directory',dirName,'created.')
    except FileExistsError:
        print("Directory",dirName,"already exists.")

    try:
        osp.mkdir(subDirTemp)
        print ('Directory',subDirTemp,'created.')
    except FileExistsError:
        print("Directory",subDirTemp,"already exists.")

    try:
        osp.mkdir(irDir)
        print ('Directory',irDir,'created.')
    except FileExistsError:
        print("Directory",irDir,"already exists.")

    try:
        osp.mkdir(nrDir)
        print ('Directory',nrDir,'created.')
    except FileExistsError:
        print("Directory",nrDir,"already exists.")
        
        
    try:
        osp.mkdir(kozDir)
        print ('Directory',kozDir,'created.')
    except FileExistsError:
        print("Directory",kozDir,"already exists.")    
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
    
    
    #---------------------------Test code below----------------------------------
    
    temp_dir = glob.glob('{}{}/{}/{}'.format(destinPath,dat,sInt,sTemp))
    res_particles = glob.glob('{}{}*/{}/{}/Particles_in_resonance_{}.txt'.format(destinPath,dat,sInt,sTemp,ST))
    
  
    
    
    #files with resonant and nonresonant particles DO NOT include the Sun. Therefore zero indexed but shifted down one
    
    for i in res_particles:
        data = []
        data_file = open(i, 'r')
        for j in data_file.readlines():
            data.append(int(j)+1)
    
    
  
    for part in range(len(sim.particles)-1, 4, -1):  
        if part not in data:
            sim.remove(part)
            print('Particles {} removed'.format(part))        
    
    #print(sim.N)    
   
        
    
    IT = shortTime          #this is the short integration run time.
    ET = ST + IT
    Ntotal = sim.N - 1 #number of objects without the sun
    npart = sim.N - sim.N_active # number of test particles
    Nout = 1000
   
   
    
    hashesParticles = np.zeros(sim.N,dtype="uint32")
    sim.serialize_particle_data(hash=hashesParticles)
    #print(hashesParticles)

    print("now running kozai short integration from {} with {} particles + giant planets + sun".format(sim.t, npart))

    intTimes = np.linspace(ST, ET, Nout)
    #print(ST)
    #print(ET)

    ### ------------ where we are going to store values ----------------- #####
    # these are 2D matrices that store the values for each particle, for each timestep
    # made them have 9999 so it's noticeable where particles escaped (values didn't store)

    l = np.zeros((Ntotal,Nout))
    p = np.zeros((Ntotal,Nout))
    a = np.ones((Ntotal,Nout))*9999
    e = np.ones((Ntotal,Nout))*9999
    lasc_node = np.ones((Ntotal,Nout))*9999
    arg_peri = np.zeros((Ntotal,Nout))
    t_anom = np.ones((Ntotal,Nout))*9999
    incl = np.ones((Ntotal,Nout))*9999
    phi = np.zeros((Ntotal,Nout))
    M_anom = np.ones((Ntotal,Nout))*9999
    xvals = np.ones((Ntotal,Nout))*9999
    yvals = np.ones((Ntotal,Nout))*9999
    timeArray = np.ones(Nout)*9999

    mln_arr = []
    mlp_arr = []
    pj_arr = []
    a_arr = []
    e_arr = []
    inc_arr = []
    lasc_node_arr = []
    arg_peri_arr = []
    t_anom_arr = []
    M_anom_arr = []
    x_arr = []
    y_arr = []
    
    """
    print('starting for loop')
    #print(sim.exit_max_distance)
    for i,times in enumerate(intTimes):
        while sim.t < times:
            print(sim.t, times)
            try:
                sim.integrate(times, exact_finish_time = 0)
                #print('in try function:'+ str(times))
            except rebound.Escape as error:
                for part in range(sim.N):
                    psim = sim.particles[part]
                    dist = psim.x*psim.x + psim.y*psim.y + psim.z*psim.z
                    #if dist > sim.exit_max_distance**2:
                        #index = psim.hash
    #                     print("particle with hash {} escaped".format(index))
                #sim.remove(hash=index)
        for j, j_hash in enumerate(hashesParticles[1:]):
    #         print("j: {} ; hash: {}".format(j, j_hash))
            try: 
                ps = sim.particles[h(c_uint32(j_hash))]
                l[j][i] = ps.calculate_orbit().l
                p[j][i] = ps.calculate_orbit().pomega
                a[j][i] = ps.calculate_orbit().a
                e[j][i] = ps.calculate_orbit().e
                incl[j][i] = ps.calculate_orbit().inc
                lasc_node[j][i] = ps.calculate_orbit().Omega 
                arg_peri[j][i] = ps.calculate_orbit().omega
                t_anom[j][i] = ps.calculate_orbit().f
                M_anom[j][i] = ps.calculate_orbit().M
                xvals[j][i] = ps.x
                yvals[j][i] = ps.y
            except rebound.ParticleNotFound as error: 
                # since particles escaping as we store/integrate
                pass
    #             print("idk {}".format(error))
    
    """
    
    
    
    for i,times in enumerate(intTimes):
        #print(times)
        sim.integrate(times, exact_finish_time = 0)
        os = sim.calculate_orbits()
        for j in range(sim.N-1):
        
            l[j][i] = os[j].l
            p[j][i] = os[j].pomega
            a[j][i] = os[j].a
            e[j][i] = os[j].e
            incl[j][i] = os[j].inc
            lasc_node[j][i] = os[j].Omega 
            arg_peri[j][i] = os[j].omega 
            #lon_peri[j][i] = os[j].pomega % (2*np.pi)
            t_anom[j][i] = os[j].f
            M_anom[j][i] = os[j].M
            xvals[j][i] = sim.particles[j].x
            yvals[j][i] = sim.particles[j].y
            #zvals[j][i] = sim.particles[j].z
            
            #renaming values
            mlp = l[j][i]
            pj = p[j][i]
            mln = l[3][i]
            sem = a[j][i]
            ecc = e[j][i]
            inc = incl[j][i]
            lan = lasc_node[j][i]
            ap = arg_peri[j][i]
            ta = t_anom[j][i]
            ma = M_anom[j][i]
            x = xvals[j][i]
            y = yvals[j][i]


            #appending to cleaned up arrays
            mln_arr.append(mln)
            mlp_arr.append(mlp)
            pj_arr.append(pj)
            a_arr.append(sem)
            e_arr.append(ecc)
            inc_arr.append(inc)
            lasc_node_arr.append(lan)
            arg_peri_arr.append(ap)
            t_anom_arr.append(ta)
            M_anom_arr.append(ma)
            x_arr.append(x)
            y_arr.append(y)

            phi_temp = 3.*mlp - 2.*mln - pj   
            phi[j][i] = phi_temp%(2*np.pi)
        
   
    #print("done: after short int {} particles left".format(sim.N))



    #change paths to wherever you want the images saved to
    #Testing for 3:2 MMR and Kozai 

    resonant_particles = []
    nonresonant_particles = []   
    kozai_particles = []
    nonkozai_particles = []
    
    for i in range(sim.N-1):
        kozai = arg_peri[i]*180/np.pi
        if ((all(kozai < 175) and all(kozai > 5)) or (all(kozai < 355) and all(kozai > 185))):
            kozai_particles.append(i)
        else:
            nonkozai_particles.append(i)
    
    
    
    with open("{}/Particles_in_Kozai_{}.txt".format(kozDir, ST), "w") as my_file:                               
        for i in kozai_particles:
            my_file.write(str(i)+"\n")
    with open("{}/Particles_not_in_Kozai_{}.txt".format(kozDir, ST), "w") as my_file:                               
        for i in nonkozai_particles:
            my_file.write(str(i)+"\n")
    
    
    #e and i values for ei plot of kozai particles
    koz_plot_data = []
    nonkoz_plot_data = []    

    ek = []
    ik = []
    for i in e[kozai_particles]:
        ek.append(i[0])
    for j in incl[kozai_particles]:
        ik.append(j[0])
    koz_plot_data.append(ek) 
    koz_plot_data.append(ik)  
    np.savetxt('{}/koz_plot_data.txt'.format(kozDir), koz_plot_data, fmt = '%s')
    
    
    enk = []
    ink = []
    
    for j in e[nonkozai_particles]:
        enk.append(j[0])
    for k in incl[nonkozai_particles]:
        ink.append(k[0])
    nonkoz_plot_data.append(enk) 
    nonkoz_plot_data.append(ink) 
    np.savetxt('{}/nonkoz_plot_data.txt'.format(kozDir), nonkoz_plot_data, fmt = '%s')

    
    #omega values for Kozai libration plots
    ok = []
    for l in arg_peri[kozai_particles]:
        ok.append(l)
    np.savetxt('{}/omega_koz_data.txt'.format(kozDir), ok, fmt = '%s')
    onk = []
    for l in arg_peri[nonkozai_particles]:
        onk.append(l)
    np.savetxt('{}/omega_nonkoz_data.txt'.format(kozDir), onk, fmt = '%s')
     
     
    print('DONE!')    
    
    
    
    return sim
