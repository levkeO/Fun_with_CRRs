"""

run as python3 CRR_shape.py [path] [file name] [lag] [temperature] [number of frames] [number of particles]
"""

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
lag = int(sys.argv[3]) # time lag max dynamic heterogeneity in frames for chosen file
rho = 1.4                       # number density for periodic boundaries
numFrames = int(sys.argv[5])
numPart = int(sys.argv[6])                 # number of particles
numFast = int(numPart*0.10)
path2 = sys.argv[1]
filexyz = sys.argv[2]           # coordinate file from command line
L  = (numPart / rho)**(1/3)
allCoords = nf.readCoords(path2+filexyz, numFrames,numPart)
counter =0
outFile ='fastPart_'+ filexyz[:-4] + '_temp_tau_a.xyz'
print(L)
def set_cell(frame, data):
        """
        Modifier to set cell of xyz-files
        """
        with data.cell_:
                data.cell_[:,0] = [L, 0., 0.]
                data.cell_[:,1] = [0., L, 0.]
                data.cell_[:,2] = [0., 0., L]
                #cell origin
                data.cell_[:,3] = [0,  0  ,  0]
                #set periodic boundary conditions
                data.cell_.pbc = (True, True, True)


def partID(frame,data):
        """
        Assigns ID to every particle
        """
        data.particles_.create_property('fast', data=fast)



sLargest = []
s2Largest = []
numClust = []
sum5 = []
writeFile = 1
if writeFile ==1:
	outFile ='results/CRRs_'+ filexyz[:-4] + '_new.xyz'
else:
	outFile = 'tmp.test'
frameCount = 0
numNeighTot =[]
numNeighCl1 = []
mobility = []
clusdistr = []
with open(outFile,'w') as outFile:
	for frame in range(lag,numFrames,1):
		node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
		node.modifiers.append(set_cell)
		dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-lag,frame,L) for particle in range(numPart)])
		mobility.append(dist)
		fastPart = dist.argsort()[-numFast:]
		fast = np.zeros(numPart)
		for particle in range(numPart):
			if particle in fastPart:
				fast[particle] = 1
		node.modifiers.append(partID)
		node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
		node.modifiers.append(ClusterAnalysisModifier(cutoff = cutoff,sort_by_size = True,only_selected = True,unwrap_particles=True))	
		data = node.compute(frame-int(lag))
		cluster_sizes = np.bincount(data.particles['Cluster'])
		sLargest.append(cluster_sizes[1])
		s2Largest.append(cluster_sizes[2])
		numClust.append(len(cluster_sizes))
		sum5.append(sum(cluster_sizes[1:6]))
		clusdistr.append(cluster_sizes[1:])
		if writeFile == 1:
			outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frameCount))
			frameCount +=1
			for particle in range(numPart):
				if particle in fastPart:
					outFile.write('A {} {} {} {}\n'.format(data.particles['Cluster'].array[particle],allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
				else:
					outFile.write('B {} {} {} {}\n'.format(1000,allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
		#Clusters:
		node.modifiers.append(ExpressionSelectionModifier(expression = 'Cluster >0 '))
		node.modifiers.append(InvertSelectionModifier())
		node.modifiers.append(DeleteSelectedModifier())
		data = node.compute(frame-lag)
		property_Neigh(data,cutoff)
		numNeigh = data.particles["neighCut"].array
		numNeighTot.append(numNeigh)
		largeClust =(cluster_sizes[cluster_sizes>=5]).argmin()
		boolExp = 'Cluster <=' + str(largeClust)
		node.modifiers.append(ExpressionSelectionModifier(expression = boolExp))
		node.modifiers.append(InvertSelectionModifier())
		node.modifiers.append(DeleteSelectedModifier())
		data = node.compute(frame)
		property_Neigh(data,cutoff)
		numNeigh = data.particles["neighCut"].array
		numNeighCl1.append(np.array(numNeigh))
numNeighTot=np.array(numNeighTot).flatten()
numNeighCl1=np.concatenate(numNeighCl1).ravel()
clusdistr=np.concatenate(clusdistr).ravel()
mobility=np.array(mobility).flatten()
T = sys.argv[4]
print(numNeighTot)
np.savetxt('results/T'+T+'_neighbours_all_lag_'+sys.argv[3]+'.txt',numNeighTot)
np.savetxt('results/T'+T+'_neighbours_larger5_lag_'+sys.argv[3]+'.txt',numNeighCl1)
np.savetxt('results/T'+T+'_mobility_lag_'+sys.argv[3]+'.txt',mobility)
np.savetxt('results/T'+T+'_clusDistr_lag_'+sys.argv[3]+'.txt',clusdistr)

