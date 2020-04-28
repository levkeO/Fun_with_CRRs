from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
import numpy as np
import pylab as pl
import sys                                  # to be able to load the file from the command line
path='/home/xn18583/Simulations/glass_KA21/facilitation/'
sys.path.append(path)
from numba import njit, config, __version__
from numba.extending import overload
import facil_module as nf
#one file that writes all the fast particles and analysis the clusters --> one new property: clusterId (=0 if not in 10% fastest

# Variables to change  and load command line arguments:
rho = 1.4                       # number density for periodic boundaries
numFrames = 512
numPart = 10002			# number of particles
numFast = int(numPart/10)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2] 		# coordinate file from command line
side  = (numPart / rho)**(1/3)
node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])

allCoords = nf.readCoords(path2+filexyz, numFrames,numPart)
counter =0
outFile ='fastPart_'+ filexyz[:-4] + '_temp.xyz'



def partID(frame,data):
	"""
	Assigns ID to every particle
	"""
	
	dist =np.array([nf.squareDist(allCoords[:,particle,:],0,frame,side) for particle in range(numPart)])
	fastPart = dist.argsort()[:numFast]
	ID = np.array(range(data.particles.count))
	fast = np.zeros(data.particles.count)
	for particle in ID:
		if particle in fastPart:
			fast[particle] = 1
	data.particles_.create_property('fast', data=fast)



sLargest = []
s2Largest = []
numClust = []
sum5 = []
with open(outFile,'w') as outFile:
	for frame in range(1,numFrames,10):
		node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
		dist =np.array([nf.squareDist(allCoords[:,particle,:],0,frame,side) for particle in range(numPart)])
		fastPart = dist.argsort()[:numFast]
		node.modifiers.append(partID)
		node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
		#node.modifiers.append(DeleteSelectedModifier())
		node.modifiers.append(ClusterAnalysisModifier(cutoff = 1.5,sort_by_size = True,only_selected = True))
		data = node.compute(frame)
		#print(np.count_nonzero(data.particles.selection))
		#print(data.particles.count)
		cluster_sizes = np.bincount(data.particles['Cluster'])
		sLargest.append(cluster_sizes[1])
		s2Largest.append(cluster_sizes[2])
		numClust.append(len(cluster_sizes))
		sum5.append(sum(cluster_sizes[1:6]))
		#print('number of clusters: ',len(cluster_sizes))
		#print('largest 5 clusters: ',cluster_sizes)
		outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frame))
		for particle in range(numPart):
			if particle in fastPart:
				outFile.write('A {} {} {} {}\n'.format(data.particles['Cluster'].array[particle],allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
			else:
				outFile.write('B {} {} {} {}\n'.format(1000,allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
pl.plot(numClust,'o')
pl.ylabel('numCLust')
pl.figure()
pl.plot(sLargest,'o')
pl.plot(s2Largest,'o')
print(np.array(numClust).argmin(),np.array(sLargest).argmax())
print('number of cluster: ',np.array(numClust).min(),'\nlargest cluster: ',np.array(sLargest).max(),'\n2nd largest: ',max(s2Largest),'\nSum of the 5 largest clusters:',max(sum5))
pl.show()

