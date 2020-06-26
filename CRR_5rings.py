import numpy as np
import pylab as pl
import sys
from numba import njit, config, __version__
from numba.extending import overload
from collections import Counter
# now per frame and check why there are so many!

file7A =  '../glass_KA21/T0.5/TCC/T0.500_N10002_KA21.xyz.rcAA2.rcAB2.rcBB2.Vor1.fc0.82.PBCs1.clusts_sp5c'
fileXYZ =  'CRRs_T0.500_N10002_KA21_new.xyz'
outXYZ = 'T0.500_N10002_KA21_5_memberedRings_test.xyz'
def load7A(file7A):
	IDs = {}
	with open(file7A, 'r') as readFile:
		for lines in readFile:
			line = lines.split()
			if line[0] == 'Frame':
				frame = int(line[2])
				print(frame)
				IDs[frame] = []
			else:
				for col in range(7):
					IDs[frame].append(float(line[col]))
	return IDs


IDs = load7A(file7A)
sevenA_pop = []
CRR = []
numFrames = 512
def writexyz(fileXYZ,outFile):
	with open(fileXYZ, 'r') as readFile, open(outXYZ, 'w') as writeFile:
		frame = -1
		for lines in readFile:
			line = lines.split()
			if len(line) ==1:
				frame +=1
				particleCounter = 0
				sevenA = Counter(IDs[frame])
				writeFile.write(lines)
			elif not line[0] == 'Atoms.':
				writeFile.write('{} {} {} {} {} {}\n '.format(line[0],sevenA[particleCounter],line[1],line[2],line[3],line[4]))
				sevenA_pop.append(sevenA[particleCounter])
				if line[0] == 'A':
					CRR.append(1)
				else:
					CRR.append(0)
				particleCounter+=1
			else:
				writeFile.write(lines)

writexyz(fileXYZ,outXYZ)
print(CRR[:20])
print(sevenA_pop[:20])
np.savetxt('CRRs_T0.500_N10002_KA21.txt',np.array(CRR))
np.savetxt('sevenAs_T0.500_N10002_KA21.txt',np.array(sevenA_pop))


