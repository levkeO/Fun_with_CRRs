"""
Calculate the percentage of particles in CRRs for particles in excitations and for all particles
and writes and xyzFile with particle type 1 if in CRR and 0 otherwise

"""
import numpy as np
import pylab as pl
import sys
import pandas as pd
from numba import njit, config, __version__
from numba.extending import overload


fa_results=pd.read_csv('/home/xn18583/Simulations/glass_KA21/facilitation/facilitation/excitation_results_Trestart_T0.55_N10002_NVT_step_1LJ_startFrame100.csv')#            '../glass_KA21/facilitation/facilitation/excitation_results_T0.55_restart_N10002_NVT_step_1LJ_startFrame100.csv'))
CRR_file =   'results/CRRs_T0.55_N10002_NVT_step_50LJ_startFrame100.xyz'
lag = 19
numPart = 10002
numFrames = 1001-lag
numReFiles = 50

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
				allCoords[frame][particleCounter,0] =splitL[2]
				allCoords[frame][particleCounter,1] =splitL[3]
				allCoords[frame][particleCounter,2] =splitL[4]
				particleCounter+=1
				if splitL[0] == 'A': CRR.append(1)
				if splitL[0] == 'B': CRR.append(0)
	return allCoords, np.reshape(CRR,(-1,numPart))



allCoords,CRR = readCoords(CRR_file, numFrames, numPart)
avFrames = 5
aver = np.zeros((numReFiles,10002))
excis = np.zeros((numReFiles,10002))
exFrames = []
for index,key in enumerate(fa_results):
	if not key == 'Unnamed: 0':
		intkey = int(key)
		#print(intkey,index)
		exInd = np.array(fa_results[key][0][1:-1].split(',')).astype(int)
		#print(exInd)
		excis[index-1,exInd]=1
		startFrame = int(intkey/1000/50)
		print(startFrame)
		exFrames.append(startFrame-lag)
		aver[index-1,:] = CRR[startFrame-lag,:]
		#for frame in range(startFrame-avFrames,startFrame+avFrames):
		#	aver[index-1,:] += CRR[frame-lag,:]/(2*avFrames)


for i in range(numReFiles):
	#print(sum(excis[:,i]))
	print(aver[i,excis[i,:].astype(int)==1].mean(),aver[i,:].mean())

numFrames = 10002
outFile = 'results/facilitation_CRR_T0.55.xyz'
with open (outFile, 'w') as outFile:
	for frame in range(numReFiles):
		outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frame))
		for particle in range(numPart):
			outFile.write('{} {} {} {} {}\n'.format(CRR[exFrames[frame],particle],excis[frame,particle],allCoords[exFrames[frame]][particle,0],allCoords[exFrames[frame]][particle,1],allCoords[exFrames[frame]][particle,2]))

