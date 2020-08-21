:! python3 % CRRs_T0.5_N10002_NVT_step_1000LJ_startFrame100_forward.xyz 0.5 100 55 26 10002 NVT 500000"""
Calculate the percentage of particles in CRRs for particles 0.0
in excitations and for all particles
and writes and xyzFile with particle type 1 if in CRR and 0 otherwise
run as:
python3 combine_CRR_fac_forward.py [CRR file] [Temperature] [start frame] [lag (min) cluster size] [number of files rerun] [number of particles] [ensemble: NVT or NVE] [scale from step to correspondng frame] 
"""
import numpy as np
import pylab as pl
import sys
import pandas as pd
from numba import njit, config, __version__
from numba.extending import overload

T = sys.argv[2]
runFrame = sys.argv[3]
fa_results=pd.read_csv('/home/xn18583/Simulations/glass_KA21/facilitation/facilitation/excitation_results_restart_T'+T+'_N10002_NVT_step_1LJ_startFrame'+runFrame+'.csv')#            '../glass_KA21/facilitation/facilitation/excitation_results_T0.55_restart_N10002_NVT_step_1LJ_startFrame100.csv'))
CRR_file =   'results/' + sys.argv[1]
lag = int(sys.argv[4])
numPart = int(sys.argv[6])
numFrames = 1001-lag
numReFiles = int(sys.argv[5])
ensemble = sys.argv[7]
scale = int(sys.argv[8]) # scaling of simulation steps to frames

def readCoords(filexyz, numFrames, numPart):
	"""
	Reads data from an xyz file
	Args:
		filexyz(string): name of the xyz file to read
		numFrames (int): number of frames in file
		numPart (int): number of particles
	Return:
		allCoords (list of list) for each frame a list of all 
		particles consisting of list of all three coordinates
		for each particle (x,y,z)
        
	"""

	frame = -1
	allCoords = np.zeros((numFrames,numPart,3))
	CRR = []
	with open(filexyz, 'r') as readFile:
		for line in readFile:
			splitL = line.split()
			if len(splitL) ==1:
				frame +=1
				if frame == numFrames:
					break
				particleCounter = 0
			elif not splitL[0] == 'Atoms.':
				allCoords[frame][particleCounter,0] =splitL[1]
				allCoords[frame][particleCounter,1] =splitL[2]
				allCoords[frame][particleCounter,2] =splitL[3]
				particleCounter+=1
				if splitL[0] == 'A': CRR.append(1)
				if splitL[0] == 'B': CRR.append(0)
	return allCoords, np.reshape(CRR,(-1,numPart))



allCoords,CRR = readCoords(CRR_file, numFrames, numPart)
#avFrames = 5
aver = np.zeros((numReFiles,numPart))
excis = np.zeros((numReFiles,numPart))
exFrames = []
startFrame = []
for index,key in enumerate(fa_results):
	print(key)
	if not key == 'Unnamed: 0':
		intkey = int(key)
		startFrame.append(int(intkey/scale))
		print('keys',intkey/scale,len(fa_results[key][0][1:-1].split(',')))
		exFrames.append(startFrame)
		exInd = np.array(fa_results[key][0][1:-1].split(',')).astype(int)
startFrame = np.array(startFrame)
startFrame.sort()
#print('Hello startFrame: ',startFrame)
for frame in range(numReFiles-3):
	exInd = np.array(fa_results[str(startFrame[frame]*scale)][0][1:-1].split(',')).astype(int)
	excis[frame,exInd]=1
	print('sum exis',frame,sum(excis[frame,:]))
		
#print(index-1,CRR[startFrame,:].max())
#for frame in range(startFrame-avFrames,startFrame+avFrames):
#	aver[index-1,:] += CRR[frame-lag,:]/(2*avFrames)

percen = []
for i in range(numReFiles-3):
	print(i,startFrame[i])
	percen.append(CRR[startFrame[i],excis[i,:].astype(int)==1].mean())
	print(startFrame[i],'prob CRR in exci: ', CRR[startFrame[i],excis[i,:].astype(int)==1].mean())
	print(sum(CRR[startFrame[i],:]),sum(excis[i,:]))
	print(sum(CRR[startFrame[i],excis[i,:].astype(int)==1]), sum(excis[i,:].astype(int)))
outFile = 'results/facilitation_CRR_T'+T+'_start'+runFrame+'_'+ensemble+'.xyz'
print(outFile)
print(percen)
np.savetxt('results/percentage_exci_in_CRR_T'+T+'_start'+runFrame+'_'+ensemble+'.txt',percen)
with open (outFile, 'w') as outFile:
	for frame in range(numReFiles-3):
		outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frame))
		for particle in range(numPart):
			outFile.write('{} {} {} {} {}\n'.format(CRR[startFrame[frame],particle],excis[frame,particle],allCoords[startFrame[frame]][particle,0],allCoords[startFrame[frame]][particle,1],allCoords[startFrame[frame]][particle,2]))

