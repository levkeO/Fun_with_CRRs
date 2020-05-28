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
from ovito.modifiers import UnwrapTrajectoriesModifier
sys.path.append(path)
from numba import njit, config, __version__
from numba.extending import overload
import facil_module as nf
L = 19.25985167448
cutoff= 1.3
from numpy.linalg import eig
import sys


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


def gyration_tensor(_r, Print=False):
    N=_r.shape[0]
    # centre of mass
    r_cm=np.mean(_r,axis=0)
    # remove the centre of mass
    r=_r-r_cm
    # initialize the tensor to zero
    S=np.zeros((3,3))
    S=np.matrix(S)
    # compute the entries
    for m in range(3):
        for n in range(3):
            S[m,n]=np.sum(r[:,m]*r[:,n])/N
    # diagonalise and derive the eigenvalues L and the eigenvectors V
    LL,V=eig(S)
    # sort the eigenvalues
    L=np.real(np.sort(LL))
    Rg=np.sqrt(np.sum(LL))
    b=L[2]-0.5*(L[1]+L[2])
    c=L[1]-L[0]
    k2=(b**2+(3./4.)*c**2)/Rg**4
    if Print:
        #print (“eigenvalues “,L)
        #print (“Radius of gyration “,Rg)
        #print (“asphericity “,b)
        #print (“acilindricity “,c)
        #print (“relative shape anisotropy “,k2)
        print('hello')
    return r_cm,V,LL,Rg,b,c,k2



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
numFast = int(numPart*0.1)
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
        data.particles_.create_property('fast', data=fast)


gyr=[]
r_gyr = np.zeros(len(range(lag,numFrames)))
num1 = np.zeros(len(range(lag,numFrames)))
for frame in range(lag,lag+10,1):
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
	#Clusters:
	#node.modifiers.append(ExpressionSelectionModifier(expression = 'Cluster >0 '))
	#node.modifiers.append(InvertSelectionModifier())
	#node.modifiers.append(DeleteSelectedModifier())
	#data = node.compute(frame)
	node.modifiers.append(ExpressionSelectionModifier(expression = 'Cluster ==1'))
	#node.modifiers.append(InvertSelectionModifier())
	#node.modifiers.append(DeleteSelectedModifier())
	node.modifiers.append(set_cell)
	node.modifiers.append(UnwrapTrajectoriesModifier())
	data = node.compute(frame)
	
	coord = data.particles['Position'].array
	a,b,c,R,d,e,f=gyration_tensor(coord)
	gyr.append(R)
	#print(coord)
	avDist = np.array([0.0,0.0,0.0])
	#print(coord[3])
		
	for particle in range(1,len(coord)):
		dist = coord[particle] - coord[0]
		dist = nf.periodic_boundary(dist,L)
		avDist += dist/(len(coord)-1)
		avPos = coord[0] + avDist
		gyration = avDist[0]**2+avDist[1]**2+avDist[2]**2
	for particle in range(1,len(coord)):
		dist = coord[particle] - avPos
		dist = nf.periodic_boundary(dist,L)
		gyration+=dist[0]**2 +dist[1]**2 + dist[2]**2
	r_gyr[frame-lag] = pl.sqrt(gyration/len(coord))
	num1[frame-lag] = len(coord)
gyr=np.concatenate(gyr).ravel()
pl.plot(num1,r_gyr,'o')
pl.plot(num1,gyr,'x')

T = sys.argv[4]
print(T)
