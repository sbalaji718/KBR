**Purpose**
We use the python package Rebound to fill phase space by placing the four giant planets and test particles within certain ranges and integrate up 4.5Gyr. 

**Instructions/Notes**
Rather than run one long integration with thousands of particles, we use the multiprocessing package to run 100+ simulations simultaneously. The "code" folder contains two .py files that are used to run the simulation ensemble. "Integrations.py" holds all the functions necessary for running the simulation while migrate.py is the run file. 

Migrate.py is where the ranges for the initial conditions and the integration timescales are set. Also has number of cores that run parallel integrations.

For the simulation referenced in the paper, we copied the "code" folder 4 times to run a total of 158 integrations across 4 nodes simultaneously.

"planetParamsCartesianAU_yr.txt" contains the initial conditions for the four giant planets. The function setupPlanetBinary reads the initial conditions from the textfile and creates a .bin file using those values. This helps with reproducability across different machines. 

