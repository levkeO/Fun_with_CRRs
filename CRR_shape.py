from ovito import dataset
import numpy as np
from ovito.vis import *
from ovito.io import import_file
from ovito.modifiers import *
import pylab as pl
from ovito.data import NearestNeighborFinder
from ovito.data import CutoffNeighborFinder
import sys
path='/home/xn18583/Simulations/glass_KA21/facilitation/'
sys.path.append(path)
from numba import njit, config, __version__
from numba.extending import overload
import facil_module as nf
L = 19.25985167448
cutoff= 1.3
def property_Neigh(data,cutoff):
        """
        Appends the number of neigbours within a cutoff to the particle properties. Name of the new property: 'neighCut'
        data: ovito data to which to add the property
        cutoff: cutoff for defining neighbours
        """
        neighArray = pl.zeros(data.particles.count)
        finder = CutoffNeighborFinder(cutoff, data)
        for index in range(data.particles.count):
                numNeig=0
                for neigh in finder.find(index):
                        neighArray[index]+=1
        data.particles_.create_property('neighCut',data=neighArray)


filexyz = sys.argv[1]+sys.argv[2]
lag = 20#argv[3] # time lag max dynamic heterogeneity in frames for chosen file

rho = 1.4                       # number density for periodic boundaries
numFrames = 350
numPart = 10002                 # number of particles
numFast = int(numPart/10)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2]           # coordinate file from command line
side  = (numPart / rho)**(1/3)
#narode = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])

allCoords = nf.readCoords(path2+filexyz, numFrames,numPart)
counter =0
outFile ='fastPart_'+ filexyz[:-4] + '_temp.xyz'



def partID(frame,data):
        """
        Assigns ID to every particle
        """

        #dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-lag,frame,side) for particle in range(numPart)])
        #fastPart = dist.argsort()[:numFast]
        #ID = np.array(range(data.particles.count))
        #fast = np.zeros(data.particles.count)
        #for particle in ID:
        #        if particle in fastPart:
        #                fast[particle] = 1
        data.particles_.create_property('fast', data=fast)



sLargest = []
s2Largest = []
numClust = []
sum5 = []
writeFile = 1
if writeFile ==1:
	outFile ='CRRs_'+ filexyz[:-4] + '_test.xyz'
else:
	outFile = 'tmp.test'
frameCount = 0
numNeighTot =[]
numNeighCl1 = []
r_gyr = np.zeros(len(range(lag,numFrames)))
num1 = np.zeros(len(range(lag,numFrames)))
with open(outFile,'w') as outFile:
	for frame in range(lag,numFrames,1):
		node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
		dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-lag,frame,side) for particle in range(numPart)])
		data = node.compute(frame-int(lag))
		fastPart = dist.argsort()[:numFast]
		ID = np.array(range(data.particles.count))
		fast = np.zeros(data.particles.count)
		for particle in ID:
			if particle in fastPart:
				fast[particle] = 1
		node.modifiers.append(partID)
		node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
		node.modifiers.append(ClusterAnalysisModifier(cutoff = cutoff,sort_by_size = True,only_selected = True))	
		
		data = node.compute(frame-int(lag))
	
		property_Neigh(data,cutoff)
		numNeigh = data.particles["neighCut"].array
		numNeighTot.append(numNeigh)
		cluster_sizes = np.bincount(data.particles['Cluster'])
		#print(len(cluster_sizes))
		sLargest.append(cluster_sizes[1])
		s2Largest.append(cluster_sizes[2])
		numClust.append(len(cluster_sizes))
		sum5.append(sum(cluster_sizes[1:6]))
		
		if writeFile == 1:
			outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frameCount))
			frameCount +=1
			for particle in range(numPart):
				if particle in fastPart:
					outFile.write('A {} {} {} {}\n'.format(data.particles['Cluster'].array[particle],allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
				else:
					outFile.write('B {} {} {} {}\n'.format(1000,allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
		#Clusters:
		node.modifiers.append(ExpressionSelectionModifier(expression = 'Cluster <6'))
		node.modifiers.append(InvertSelectionModifier())
		node.modifiers.append(DeleteSelectedModifier())
		data = node.compute(frame)
		coord = data.particles['Position'].array
		avDist = np.array([0.0,0.0,0.0])
		#print(coord[3])
		
		for particle in range(1,len(coord)):
			dist = coord[particle] - coord[0]
			dist = nf.periodic_boundary(dist,L)
			avDist += dist/(len(coord)-1)
		avPos = coord[0] + avDist
		#print(avPos)
		gyration = avDist[0]**2+avDist[1]**2+avDist[2]**2
		#print(avDist)
		for particle in range(1,len(coord)):
			dist = coord[particle] - avPos
			dist = nf.periodic_boundary(dist,L)
			gyration+=dist[0]**2 +dist[1]**2 + dist[2]**2
		#print(gyration, 'gyration', len(coord))
		r_gyr[frame-lag] = pl.sqrt(gyration/len(coord))
		num1[frame-lag] = len(coord)
		property_Neigh(data,cutoff)
		numNeigh = data.particles["neighCut"].array
		numNeighCl1.append(numNeigh)
	#print(numNeighCl1[0])

		#print(data.particles.count)
# compute the fastest particles for a moving frame: frame + or - lag
# analyse cluster at intermediate frame ( once the particles are deleted I should be able to to 
# compute another frame for the same particles
pl.plot(num1,r_gyr,'o')
pl.show()
numNeighTot=np.array(numNeighTot).flatten()
numNeighCl1=np.array(numNeighCl1).flatten()
print(numNeighTot)
#np.pad(num1,(0,9002),mode='constant',constant_values=np.nan)
print(len(num1),len(r_gyr), len(numNeighTot),len(numNeighCl1))
np.savetxt('T0.48_fractal_dimension_cut1_3.txt',[num1,r_gyr])
#np.savetxt('T0.48_neighbours_cut1_3.txt',[numNeighTot])
# compute the clusters and either choose largest 1-5 or all larger than 50 -100 particles
# select cluster  and compute NN distrbution, mean nearest neighbour, radius of gyration, fractal dimension
print(np.array(sLargest).max(),np.array(s2Largest).max(),np.array(sum5).max(),np.array(numClust).max())
print(' Largest, 2nd largest, sum 5 largest, number of clusters')
print(np.array(sLargest).mean(),np.array(s2Largest).mean(),np.array(sum5).mean(),np.array(numClust).mean())
