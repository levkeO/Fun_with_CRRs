import numpy as np
import pylab as pl
import sys
path='../glass_KA21/facilitation/facilitation/'
sys.path.append(path)
import singPartDist as sp

inFile = '/home/xn18583/Downloads/pos_mW-eq2.dat'
outFile = 'test/test.xyz'
frameCounter = 0
numPart = 4000
numFrames = 201


def readCoords(filexyz, numFrames, numPart):
        """
        Reads data from an xyz file
        Args:
                filexyz(string): name of the xyz file to read
                numFrames (int): number of frames in file
                numPart (int): number of particles
        Return:
                allCoords (list of list) for each frame a list of all 
                particles consisting of list of all three coordinates
                for each particle (x,y,z)
        
        """

        frame = -1
        allCoords = np.zeros((numFrames,numPart,4))
        with open(filexyz, 'r') as readFile:
                for line in readFile:
                        splitL = line.split()
                        if len(splitL)>1 and splitL[1] =='ATOMS':
                                frame +=1
                                particleCounter = 0
                        if len(splitL) ==5:
                                allCoords[frame][particleCounter,0] =splitL[2]
                                allCoords[frame][particleCounter,1] =splitL[3]
                                allCoords[frame][particleCounter,2] =splitL[4]
                                allCoords[frame][particleCounter,3] =splitL[0]
                                particleCounter+=1
        return allCoords

allCoords = readCoords(inFile,numFrames,numPart)
with  open(outFile,'w') as outF:
	for frame in range(numFrames):
		outF.write('{} \n'.format(numPart))
		outF.write('Atoms. Timestep: {}\n'.format(frame))
		sorting = np.argsort(allCoords[frame][:,3])
		coordsx = allCoords[frame][sorting,0]
		coordsy = allCoords[frame][sorting,1]
		coordsz = allCoords[frame][sorting,2]
		print(frame)
		for particle in range(numPart):
			outF.write('A {} {} {} \n'.format(coordsx[particle],coordsy[particle],coordsz[particle]))
			
