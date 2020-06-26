

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
cutoff= 1.3
from numpy.linalg import eig


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

#njit
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
numPart = int(sys.argv[6])                 # number of particles
numFast = int(numPart*0.1)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2]           # coordinate file from command line
L  = (numPart / rho)**(1/3)
print(L)
allCoords = nf.readCoords(path2+filexyz, numFrames,numPart)
counter =0
outFile ='fastPart_'+ filexyz[:-4] + '_temp.xyz'


def radGyr(coords_RG,L):
	"""
	radius of gyration
	"""
	avDist = np.array([0.0,0.0,0.0])
	for particle in range(1,len(coords_RG)):
		dist = coords_RG[particle] - coords_RG[0]
		dist = nf.periodic_boundary(dist,L)
		avDist += dist/(len(coords_RG)-1)
		avPos = coords_RG[0] + avDist
		avPos = nf.periodic_boundary(avPos,L)
		gyration = avDist[0]**2+avDist[1]**2+avDist[2]**2
	for particle in range(1,len(coords_RG)):
		dist = coords_RG[particle] - avPos
		dist = nf.periodic_boundary(dist,L)
		gyration+=dist[0]**2 +dist[1]**2 + dist[2]**2

	return gyration
	
def partID(frame,data):
        """
        Assigns ID to every particle
        """
        data.particles_.create_property('fast', data=fast)


gyr=[]
num_gyr =[]
r_gyr = []# np.zeros(len(range(lag,numFrames)))
num1 = []#np.zeros(len(range(lag,numFrames)))
ds = []
es = []
fs =[]
clusID =[]
clusCount = []
for frame in range(lag,numFrames,1):
	print('frame: ',frame)
	node = import_file(sys.argv[1]+sys.argv[2],multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
	node.modifiers.append(set_cell)
	dist =np.array([nf.squareDist(allCoords[:,particle,:],frame-lag,frame,L) for particle in range(numPart)])
	data = node.compute(frame)
	fastPart = dist.argsort()[-numFast:]
	ID = np.array(range(data.particles.count))
	fast = np.zeros(data.particles.count)
	node.modifiers.append(partID)
	node.modifiers.append(ExpressionSelectionModifier(expression = 'ParticleType ==1 '))
	for particle in ID:
		if particle in fastPart:
			fast[particle] = 1
	node.modifiers.append(partID)
	node.modifiers.append(ExpressionSelectionModifier(expression = 'fast ==1 '))
	node.modifiers.append(ClusterAnalysisModifier(cutoff = cutoff,sort_by_size = True,only_selected = True))	
	data = node.compute(frame)
	#print('number of clusters: ',len(np.bincount(data.particles['Cluster'])))
	largeClust = np.where(np.bincount(data.particles['Cluster'])>=5)[0]
	#print(largeClust[1:])
	for clust in largeClust[1:]:
		boolStr = 'Cluster =='+str(clust)
		node.modifiers.append(ExpressionSelectionModifier(expression =boolStr))
		data=node.compute(frame)
		print('number of coordinates in cluster number ',clust, 'is ', sum(data.particles.selection.array))
		coords_RG = data.particles['Position'].array
		coords_RG = coords_RG[np.nonzero(data.particles.selection)]
		#node.modifiers.append(UnwrapTrajectoriesModifier())
		#data = node.compute(frame)
		#print('cluster: ',clust,'numPart',sum(data.particles.selection.array))
		#coord = data.particles['Position'].array
		#coord = coord[np.nonzero(data.particles.selection)]
		#a,b,c,R,d,e,f=gyration_tensor(coord)
		#gyr.append(R)
		#ds.append(d)
		#es.append(e)
		#fs.append(f)
		#node.modifiers.append(WrapPeriodicImagesModifier())
		clusCount.append(sum(data.particles.selection.array))
		clusID.append(clust)
		num_gyr.append(len(coords_RG))
		gyration = radGyr(coords_RG,L)
		r_gyr.append(pl.sqrt(gyration/len(coords_RG)))
#gyr=np.array(gyr).flatten()
T = sys.argv[4]
pl.savetxt('r_gyr_noTen_num_T'+T+'_lag'+str(lag)+'_N_'+str(numPart)+'.txt',np.real([num_gyr,r_gyr,clusID,clusCount]))
