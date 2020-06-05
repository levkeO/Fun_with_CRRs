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
numFrames = int(sys.argv[3])
numPart = 10002			# number of particles
numFast = int(numPart/10)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2] 		# coordinate file from command line
side  = (numPart / rho)**(1/3)
node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])

allCoords = nf.readCoords(path2+filexyz, numFrames,numPart)
counter =0

L = 19.25985167448

def set_cell(frame, data):
        """
        Modifier to set cell of xyz-files
        """
        with data.cell_:
                data.cell_[:,0] = [L, 0., 0.]
                data.cell_[:,1] = [0., L, 0.]
                data.cell_[:,2] = [0., 0., L]
                #cell origin
                data.cell_[:,3] = [-L/2,  -L/2  ,  -L/2]
                #set periodic boundary conditions
                data.cell_.pbc = (True, True, True)




def partID(frame,data):
	"""
	Assigns ID to every particle
	"""
	
	data.particles_.create_property('fast', data=fast)



writeFile = 0
if writeFile == 0:
	outFi = 'temp.txt'
else:
	outFi ='fastPart_'+ filexyz[:-4] + '_temp.xyz'

t_numClmin = []
t_max_largest = []
t_5max = []
startFrames = range(0,400,50)
for startFrame in  startFrames:
	sLargest = []
	s2Largest = []
	numClust = []
	sum5 = []
	print('startFrame: ',startFrame)
	with open(outFi,'w') as outFile:
		for frame in range(startFrame+1,numFrames,1):
			node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
			node.modifiers.append(set_cell)
			data = node.compute(frame)
			dist =np.array([nf.squareDist(allCoords[:,particle,:],startFrame,frame,side) for particle in range(numPart)])
			fastPart = dist.argsort()[:numFast]
			ID = np.array(range(data.particles.count))
			fast = np.zeros(data.particles.count)
			for particle in ID:
				if particle in fastPart:
					fast[particle] = 1
			node.modifiers.append(partID)
			node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
			node.modifiers.append(ClusterAnalysisModifier(cutoff = 1.3,sort_by_size = True,only_selected = True))
			data = node.compute(frame)
			cluster_sizes = np.bincount(data.particles['Cluster'])
			sLargest.append(cluster_sizes[1])
			s2Largest.append(cluster_sizes[2])
			numClust.append(len(cluster_sizes))
			sum5.append(sum(cluster_sizes[1:6]))
			if not outFile ==0:
				outFile.write('{}\nAtoms. Timestep: {}\n'.format(numPart,frame))
				for particle in range(numPart):
					if particle in fastPart:
						outFile.write('A {} {} {} {}\n'.format(data.particles['Cluster'].array[particle],allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
					else:
						outFile.write('B {} {} {} {}\n'.format(1000,allCoords[frame][particle,0],allCoords[frame][particle,1],allCoords[frame][particle,2]))
	pl.figure(1)
	pl.plot(numClust,'o',label=startFrame)
	pl.ylabel('numCLust')
	pl.figure(2)
	pl.plot(sLargest,'o',label=startFrame)
	print('startFrame: ',startFrame)
	print(np.array(numClust).argmin(),np.array(sLargest).argmax())
	print('number of cluster: ',np.array(numClust).min(),'\nlargest cluster: ',np.array(sLargest).max(),'\n2nd largest: ',max(s2Largest),'\nSum of the 5 largest clusters:',max(sum5))
	t_numClmin.append(np.array(numClust).argmin())
	t_max_largest.append(np.array(sLargest).argmax())
	t_5max.append(np.array(sum5).argmax())
print('time minimum number of clusters: ', t_numClmin)
print('time maximum largest cluster: ', t_max_largest)
print('time largest 5 clusters: ',t_5max)
pl.savetxt('CRR_finder_'+filexyz[:-4]+'.txt',[startFrames,t_numClmin,t_max_largest,t_5max])
pl.legend(frameon=False)
pl.figure(1)
pl.legend(frameon=False)

pl.show()

