import numpy as np
import pylab as pl
import sys
import pandas as pd

fa_results=(pd.read_csv('../glass_KA21/facilitation/facilitation/excitation_results_T0.55_restart_N10002_NVT_step_1LJ_startFrame400.csv'))
CRR_file =   'results/CRRs_T0.52_N10002_NVT_step_100LJ_startFrame500.xyz'



def readCRR(fileXYZ):
	CRR = []
	counter = 0
	with open(fileXYZ, 'r') as readFile:
		for lines in readFile:
			line = lines.split()
			if line[0] == 'Atoms.':
				counter+=1
			if line[0] == 'A':
				CRR.append(1)
			elif line[0] == 'B':
				CRR.append(0)
	print('number of frames is: ',counter)
	print(np.reshape(CRR,(10002,-1)).shape)
	return np.reshape(CRR,(10002,-1))
CRR = readCRR(CRR_file)
print(CRR.shape[1])

avFrames = 5
aver = np.zeros((10002,8))
excis = np.zeros((10002,8))
print(len(fa_results))
for index,key in enumerate(fa_results):
	if not key == 'Unnamed: 0':
		intkey = int(key)
		#print(intkey,index)
		exInd = np.array(fa_results[key][0][1:-1].split(',')).astype(int)
		#print(exInd)
		excis[exInd,index-1]=1
		startFrame = int(intkey/1000/50)
		for frame in range(startFrame-avFrames,startFrame+avFrames):
			#print('aver: ',aver.shape,'aver[index]: ',aver[index].shape,'CRR: ',CRR.shape,'CRR[:,frame]: ',CRR[:,frame].shape)
			aver[:,index-1] += CRR[:,frame]/(2*avFrames)


for i in range(8):
	#print(sum(excis[:,i]))
	print(aver[excis[:,i].astype(int),i].mean(),aver.mean())
				

