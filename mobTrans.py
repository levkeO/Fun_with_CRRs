import numpy as np
import pylab as pl
import sys                                  # to be able to load the file from the command line
path='/home/xn18583/Simulations/glass_KA21/facilitation/'
sys.path.append(path)
from numba import njit, config, __version__
from numba.extending import overload
import facil_module as nf


cutoff = 1.3
rho = 1.4                       # number density for periodic boundaries
numFrames = int(sys.argv[3])
numPart = 10002			# number of particles
numFast = int(numPart*0.05)
path = sys.argv[1]
filexyz = sys.argv[2] 		# coordinate file from command line
side  = (numPart / rho)**(1/3)
allCoords = nf.readCoords(path+filexyz, numFrames,numPart) #better write a modifiert for ovito?

def selectRand(allCoords,numPart,delta,numFast):
	"""
	Randomly reassignes particle types in the same ratio as before (for chosen particle type)
	Ovito modifier
	"""
	randFast = []
	for frame in range(delta,numFrames,delta):
		randType = np.random.random_sample(size=numPart)
		randType= (randType<(numFast/numPart)).astype(int)
		randFast.append(np.where(randType == 1)[0])
	return randFast

def minDistPart(allCoords,ID,IDlist,frame1,frame2,L):
	"""
	Calculate the distances of one particle to a set of particle in the previous frame
	and compute the minimum. 
	"""
	distances = allCoords[frame1][ID,:]-allCoords[frame2][IDlist,:]
	for dist in range(len(distances)):
		distances[dist,:] = nf.periodic_boundary(distances[dist,:],L)
	dist1D = np.sqrt(distances[:,0]**2 + distances[:,1]**2 + distances[:,2]**2)
	return min(dist1D)



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

def selectFast(allCoords,delta,numFast,numFrames,side):
	"""
		split trajectory into bits of length delta and identify the numFast particles 
	"""
	fastPart = []
	for frame in range(delta,numFrames,delta):
		dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-delta,frame,side) for particle in range(numPart)])
		fastPart.append(dist.argsort()[-numFast:])
	return fastPart


def distFast(fastPart,refPart,allCoords,delta):
	minDists = []
	print('new!')
	for frame in range(1,len(fastPart)):
		for partID in fastPart[frame]:
			if not  partID in refPart[frame-1]:
				distMin = minDistPart(allCoords,partID,refPart[frame-1],(frame)*delta,frame*delta,side)
				minDists.append(distMin)
	return minDists
	
#start with last frame but think about middle fram

for delta in [75,100,150,200]:
	fastPart = selectFast(allCoords,delta,numFast,numFrames,side)
	randPart = selectRand(allCoords,numPart,delta,numFast)
	minDistsFast = np.array(distFast(fastPart,fastPart, allCoords,delta))
	minDistsRand = np.array(distFast(fastPart,randPart, allCoords,delta))
	pl.hist(minDistsFast,bins = 20,histtype = 'step', color = 'red')
	print('delta:',delta,'fast: ',len(minDistsFast[minDistsFast<cutoff])/len(minDistsFast))
	print('delta',delta,'random: ',len(minDistsRand[minDistsRand<cutoff])/len(minDistsRand))
