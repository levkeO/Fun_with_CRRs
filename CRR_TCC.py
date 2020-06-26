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
#T0.500_N10002_KA21.xyz.rcAA2.rcAB2.rcBB2.Vor1.fc1.PBCs1.raw_11A
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
numPart = 10002                 # number of particles
numFast = int(numPart*0.10)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2]           # coordinate file from command line
side  = (numPart / rho)**(1/3)
#narode = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])

allCoords = nf.readCoords(path2+filexyz, numFrames,numPart)
counter =0
outFile ='TCC_CRR_'+ filexyz[:-4] + '_temp.xyz'

def set_cell(frame, data):
        """
        Modifier to set cell of xyz-files
        """
        with data.cell_:
                data.cell_[:,0] = [L, 0., 0.]
                data.cell_[:,1] = [0., L, 0.]
                data.cell_[:,2] = [0., 0., L]
                #cell origin
                data.cell_[:,3] = [0,  0  ,0]
                #set periodic boundary conditions
                data.cell_.pbc = (True, True, True)


def partID(frame,data):
        """
        Assigns ID to every particle
        """
        data.particles_.create_property('fast', data=fast)



def readTCC(tccFile):
	"""
	Makes a list of lists with particle type (in 11A/ not in 11A)
	of all particles in the file
	"""
	tcc_part = {}
	counter = -1
	with open(tccFile,'r') as tcc:
		for lines in tcc:
			line = lines.split()
			if  line[0] =='frame':
				counter+=1
				tcc_part[counter]=[]
			if line[0] == 'C' or line[0] == 'D':
				tcc_part[counter].append(1)
			elif line[0] == 'A' or line[0] == 'B':
				tcc_part[counter].append(0)
	return tcc_part
tcc_part=readTCC(sys.argv[1]+'/TCC/' + sys.argv[2]+'.rcAA2.rcAB2.rcBB2.Vor1.fc1.PBCs1.raw_11A')
frameCount=0
writeFile = 0
all_CRR=[]
all_11A = []
if writeFile ==1:
	outFile ='TCC_CRRs_'+ filexyz[:-4] + '_test.xyz'
else:
	outFile = 'tmp.test'
with open(outFile,'w') as outFile:
	for frame in range(lag,numFrames,1):
		node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
		node.modifiers.append(set_cell)
		dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-lag,frame,side) for particle in range(numPart)])
		data = node.compute(frame-int(lag))
		fastPart = dist.argsort()[-numFast:]
		ID = np.array(range(data.particles.count))
		fast = np.zeros(data.particles.count)
		for particle in ID:
			if particle in fastPart:
				fast[particle] = 1
		all_CRR.append(fast)
		all_11A.append(tcc_part[frame])
		node.modifiers.append(partID)
		node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
		node.modifiers.append(ClusterAnalysisModifier(cutoff = cutoff,sort_by_size = True,only_selected = True))	
		data = node.compute(frame-int(lag))
		cluster_sizes = np.bincount(data.particles['Cluster'])
		print(cluster_sizes[:4])
		if writeFile == 1:
			outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frameCount))
			frameCount +=1
			for particle in range(numPart):
				if particle in fastPart:
					outFile.write('{} 1 {} {} {} {}\n'.format(tcc_part[frame][particle],data.particles['Cluster'].array[particle],allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
				else:
					outFile.write('{} 0 {} {} {} {}\n'.format(tcc_part[frame][particle],1000,allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))


T = sys.argv[4]
pl.savetxt('/results/all_T'+T+'_CRR_TCC_binary_lag'+sys.argv[3]+'.txt',[np.array(all_CRR).flatten(),np.array(all_11A).flatten()])
