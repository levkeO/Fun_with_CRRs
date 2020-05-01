from ovito import dataset
import numpy as np
from ovito.vis import *
from ovito.io import import_file
from ovito.modifiers import *
import pylab as pl
from ovito.data import NearestNeighborFinder
from ovito.data import CutoffNeighborFinder
import sys

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
lag = argv[3] # time lag max dynamic heterogeneity in frames for chosen file

rho = 1.4                       # number density for periodic boundaries
numFrames = 1000
numPart = 10002                 # number of particles
numFast = int(numPart/10)
print(numFast)
path2 = sys.argv[1]
filexyz = sys.argv[2]           # coordinate file from command line
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



for frame in range(numFrames
