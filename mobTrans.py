from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
import numpy as np
import pylab as pl
import sys                                  # to be able to load the file from the command line
path='/home/xn18583/Simulations/glass_KA21/facilitation/'
sys.path.append(path)




#!!!Kick ovito out, can't use nn here any because I am comparing different frames

from numba import njit, config, __version__
from numba.extending import overload
import facil_module as nf
#one file that writes all the fast particles and analysis the clusters --> one new property: clusterId (=0 if not in 10% fastest
cutoff = 1.3
# Variables to change  and load command line arguments:
rho = 1.4                       # number density for periodic boundaries
numFrames = sys.argv[3]
numPart = 10002			# number of particles
numFast = int(numPart/10)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2] 		# coordinate file from command line
side  = (numPart / rho)**(1/3)
node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
allCoords = nf.readCoords(path2+filexyz, numFrames,numPart) #better write a modifiert for ovito?

def randomParticleType(frame,data):
        """
        Randomly reassignes particle types in the same ratio as before (for chosen particle type)
        Ovito modifier
        """
        randType = np.random.random_sample(size=(data.particles.count))
        randType= (randType<(randSel/data.particles.count)).astype(int)
        data.particles_.create_property('randProp', data=randType)



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

def countNeigh(data):
        finder = CutoffNeighborFinder(cutoff, data)
        counter=0
        for index in range(data.particles.count):
                partCount =0
                for neigh in finder.find(index):
                        counter+=1
        return counter

delta = 10
for frame in range(delta,node.source.num_frames-delta):
	node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
	dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-delta,delta,frame,side) for particle in range(numPart)])
	fastPart = dist.argsort()[:numFast]
	ID = np.array(range(data.particles.count))
	fast = np.zeros(data.particles.count)
	for particle in ID:
		if particle in fastPart:
			fast[particle] = 1
	node.modifiers.append(partID)
	node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
	node.modifiers.append(DeleteSelectedModifier())
	a = countNeigh(data)
	
pl.figure()
pl.show()

