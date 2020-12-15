import rebound
import numpy as np
import random
import time 
import os as osp       
import shutil
import itertools
from rebound import hash as h
from ctypes import c_uint32



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

    sun.hash = 1
    jupiter.hash = 2
    saturn.hash = 3
    uranus.hash = 4
    neptune.hash = 5

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
        sim.add(primary = com, a=sem, e=ecc, inc= get_i(), omega=np.random.uniform(0,2*np.pi),Omega=np.random.uniform(0, 2*np.pi),M=np.random.uniform(0,2*np.pi), hash = p+6) 


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
    dat= time.strftime("%b%d%Y.%H.%M", tm)        

  
    sim_length = integration_times[-1]
    print("integration number {}; integrating for {} years".format(int_count, sim_length))

    #the following code should be set up on first use to locate and store your simulations
    sourcePath = '/Users/arceliahermosillo/Research/kboTest/code4/' 
    destinPath = '/Users/arceliahermosillo/Research/kboTest/Long_Integrations/'


    if not long_int_file:
        filename = '{}_part{}_time{}_A_{:.3f}-{:.3f}_iSig_{}_E_{:.3f}-{:.3f}_even_q_{}'.format(dat,Nparticles,totalTime,minA,maxA,14,minE,maxE,i)
    else:
        filename = long_int_file 

    # check to see if long int already exists. If so, nothing else needs to happen. If it doesn't then we continue with long integration

    # ------- Manual Snapshots --------------------------
    try:
        sim = rebound.Simulation("{}{}.bin".format(destinPath,filename))
        print("tried loading simulation archive\n")
    except:
        print("failed to load simulation archive, loading initial conditions")
        sim = setupSimulationFromArchive()
        for t in integration_times:
            sim.integrate(t, exact_finish_time=0)
            print("done: {} ; N particles: {}".format(t, sim.N))
            print("time passed: {}".format(time.time() - start_time))
            sim.simulationarchive_snapshot('{}{}.bin'.format(destinPath, filename))

        print("long integration is done")
    
    print("long Integration took {}".format(time.time() - start_time))

    return sim



def short_integration(integrationN, minA, maxA, minE, maxE, maxDistance, shortTime, fileName, filename, indexSimulation):

    start_simul_time = time.time()

    tm = time.gmtime()
    dat= time.strftime("%b%d", tm)

    #the following code should be set up on first use to locate and store your simulations

    destinPath = '/Users/arceliahermosillo/Research/kboTest/Long_Integrations/'

    longInt    = '{}{}/{}'.format(destinPath,fileName,filename)

    sa = rebound.SimulationArchive("{}.bin".format(longInt)) #names archive object 
    sim = sa[indexSimulation] ## see comment above for this 
    ST = sim.t             #(the snapshot time we're using as initial conditions)
    # doing this just for naming purposes

    sTemp    = 'TemporaryDirectory_time_{}'.format(np.round(ST))
    iRes     = 'In_Resonance'
    nRes     = 'Not_In_Resonance'
    sInt     = 'Short_Integrations_{}'.format(dat)

    mainDir    = '{}{}/'.format(destinPath,fileName)
    dirName    = '{}{}/{}'.format(destinPath,fileName,sInt)
    subDirTemp = '{}{}/{}/{}'.format(destinPath,fileName,sInt,sTemp)
    irDir      = '{}{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,iRes)
    nrDir      = '{}{}/{}/{}/{}'.format(destinPath,fileName,sInt,sTemp,nRes)

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


    IT = shortTime          #this is the short integration run time.
    ET = ST + IT
    Nout = 1000
    Ntotal = sim.N -1 #number of objects without the sun
    npart = sim.N - sim.N_active # number of test particles
    sim.exit_max_distance = maxDistance
    #sim.ri_whfast.safe_mode = 0
    #sim.ri_whfast.corrector = 11 

    # setting hashes to the particles just for now since doing 1e9 integration before particle removal and hashes were done
    ### TAKE THIS OFF LATER ############################################3

    #for partN in range(sim.N):
        #sim.particles[partN].hash = partN + 1

    ### TAKE THIS OFF LATER. JUST HAVE IT FOR THE 1E9 LONG INT THAT RAN BEFORE REMOVAL OF PARTICLES WAS DONE ###########

    print("starting short integration {} with start time {}".format(integrationN, ST))
    
    ecc = np.zeros(npart)
    ax = np.zeros(npart)
    inc = np.zeros(npart)
    
    for i in range(npart):
        ecc[i] = sim.particles[i+5].e
        ax[i] = sim.particles[i+5].a
        inc[i] = sim.particles[i+5].inc
        
        """
    arange = np.linspace(37,50)
    fig, (p1, p2) = plt.subplots(1,2, figsize = (11,5))
    p1.plot(ax, ecc, '.')
    p2.plot(ax, inc, '.')
    p1.plot(arange, 1-20/arange , '--', label = "r = 20AU")
    p1.plot(arange, 1-25/arange , '--', label = "r = 25AU")
    p1.plot(arange, 1-30/arange, '--', label = "r = 30AU")
    p1.plot(arange, 1-35/arange, '--', label = "r = 35AU")
    p1.vlines(minA,0, 1.1)
    p1.vlines(maxA, 0,1.1)
    p1.hlines(0.6, 37, 50)
    p1.set_xlim(37, 50)
    p1.set_ylim(0,1.1)
    p1.set_title("a v e; t = {}".format(ST))
    p1.set_xlabel("a [AU]")
    p1.set_ylabel("e ")
    p2.vlines(minA, 0, np.pi/2)
    p2.vlines(maxA, 0, np.pi/2)
    p2.set_xlim(37, 50)
    p2.set_ylim(0,np.pi/2)
    p2.set_title("a v i; t = {}".format(ST))
    p2.set_xlabel("a [AU]")
    p2.set_ylabel("i")
    p1.legend()
    plt.savefig('{}/ave_avi_plot'.format(subDirTemp))
"""
    # pointer to get the hashes of particles still alive

    hashesParticles = np.zeros(sim.N,dtype="uint32")
    sim.serialize_particle_data(hash=hashesParticles)
    #print(hashesParticles)

    intTimes = np.linspace(ST, ET, Nout)

    ### ------------ where we are going to store values ----------------- #####
    # these are 2D matrices that store the values for each particle, for each timestep
    # made them have 9999 (or 0 for values that are used to calculate phi) so it's noticeable where particles escaped (values didn't store)

    l = np.zeros((Ntotal,Nout))
    p = np.zeros((Ntotal,Nout))
    a = np.ones((Ntotal,Nout))*9999
    e = np.ones((Ntotal,Nout))*9999
    lasc_node = np.ones((Ntotal,Nout))*9999
    arg_peri = np.ones((Ntotal,Nout))*9999
    t_anom = np.ones((Ntotal,Nout))*9999
    incl = np.ones((Ntotal,Nout))*9999
    phi = np.zeros((Ntotal,Nout))
    M_anom = np.ones((Ntotal,Nout))*9999
    xvals = np.ones((Ntotal,Nout))*9999
    yvals = np.ones((Ntotal,Nout))*9999
    timeArray = np.ones(Nout)*9999
    peri = np.ones((Ntotal, Nout))*9999


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

    #print(sim.exit_max_distance)


    for i,times in enumerate(intTimes):
        while sim.t < times:
            try:
                #sim.ri_whfast.recalculate_coordinates_this_timestep = 1
                sim.integrate(times, exact_finish_time = 0)
            except rebound.Escape as error:
                #print(error)
                for part in range(sim.N):
                    psim = sim.particles[part]
                    dist = psim.x*psim.x + psim.y*psim.y + psim.z*psim.z
                    if dist > sim.exit_max_distance**2:
                        index = psim.hash
    #                     print("particle with hash {} escaped".format(index))
                sim.remove(hash=index)
                #sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        #sim.integrator_synchronize()
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

            peri[j][i] = sem*(1-ecc)

            phi_temp = 3.*mlp - 2.*mln - pj   
            phi[j][i] = phi_temp%(2*np.pi)

    print("done: after short int {} particles left".format(sim.N))


    resonant_particles = []
    count = 0        
    for i in range(Ntotal):
        phi_i = phi[i]
        if (all(phi[i] < 355*np.pi/180) and all(phi[i] > 5*np.pi/180)):
            #print(phi_i)
            resonant_particles.append(i)
            count +=1

            #plt.figure(figsize=(15,10))
            #plt.title('Resonant angle libration', fontsize = 24)
            #plt.xlabel('Time(years)', fontsize = 18)
            #plt.ylabel('Resonant argument (degrees)', fontsize = 18)
            #plt.scatter(intTimes,phi_i, marker = '.',s = 10)
            #plt.ylim(0, 2*np.pi)
            #plt.savefig('{}/Particle {} Phi vs Time Plot.png'.format(irDir,i))  
            #plt.clf()

    with open("{}/Particles_in_resonance_{}.txt".format(subDirTemp, np.round(ST)), "w+") as my_file:                               
        for i in resonant_particles:
             my_file.write(str(i)+"\n")



    nonresonant_particles = []
    count_n = 0        
    for j in range(Ntotal):
        phi_n = phi[j]
        if (any(phi[j] > 355*np.pi/180) and any(phi[j] < 5*np.pi/180)):
            #print(phi_n)
            nonresonant_particles.append(j)
            count_n +=1

            #plt.figure(figsize=(15,10))
            #plt.title('Resonant angle circulation', fontsize = 24)
            #plt.xlabel('Time(years)', fontsize = 18)
            #plt.ylabel('Resonant argument (degrees)', fontsize = 18)
            #plt.scatter(intTimes,phi_n, marker = '.',s = 10)
            #plt.ylim(0,2*np.pi)
            #plt.savefig('{}/Particle {} Phi vs Time Plot.png'.format(nrDir,i))  
            #plt.clf()

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


            #xsur = 29.06807239827766   
            #ysur = -7.125912195043246
            xsur = 26.85758046958696
            ysur = -13.32890006819031
            # for julian date ? 
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

    phiAmp = []

    for i in range(len(phi)):
        pamp = (np.max(phi[i]) - np.min(phi[i]))/2
        phiAmp.append(pamp)

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

    #with open('{}/{}_data_array_{}.csv'.format(subDirTemp,fileName, ST), mode = 'w') as file:
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


