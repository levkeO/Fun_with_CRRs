"""
Calculate g(r) from an xyz file. There is the option to chose one particle Typ
run as python3 gr_for_the_people.py [path to file] [xyz-file] [particle type (optional)]
plots and saves gr figure and saved r and gr as numpy files
ovito for python has to be installed
"""

from ovito.io import *
from ovito.data import *
from ovito.modifiers import *
import numpy as np
import pylab as pl
import sys
import pandas as pd
#Parameters that you might want to change:
cut = 10		# cutoff for g(r)
numBins = 250		# number of bins u=in g(r) calculation
L = 19.25985167448	# length of box
start =400#None		# first frame to calculate, if this is set to None it will calculate the entire trajectory
end =600# None		# last frame to calculate

node = import_file(sys.argv[1]+'T'+sys.argv[2]+'_N10002_NVT_step_0.1LJ_startFrame500.xyz', multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])

def gr_calc(node,cut,bins,partType = None,excitation=None,startFrame=0,endFrame=node.source.num_frames,step=1):
	"""
	Calculates the radial distribution function g(r) of one particle type or all particles (if no partType given)
	Args: 

		node (ovito data pipeline): all the data from file
		cut (float): cutoff for g(r)
		bins(int): number of bins for g(r) histogram
		partType (int or string):  desired particle type
		startFrame (int): first frame to average over
		endFrame (int): last Frame to average over
		step(int) : skipping frames
	Returns:
		r (numpy array): r values of g(r)
		gr: radial distribution function g(r)for r values
		
	"""
	# If particle type is given delete all other partcles
	if partType:
		node.modifiers.append(SelectTypeModifier(types={partType}))
		node.modifiers.append(InvertSelectionModifier())
		node.modifiers.append(DeleteSelectedModifier())
	elif excitation:
		node.modifiers.append(exciID)
		node.modifiers.append(ExpressionSelectionModifier(expression = 'exci ==1 '))
		node.modifiers.append(InvertSelectionModifier())
		node.modifiers.append(DeleteSelectedModifier())

	node.modifiers.append(set_cell)
	modifier = CoordinationAnalysisModifier(cutoff = cut, number_of_bins = bins)
	node.modifiers.append(modifier)
	rdf = np.zeros((bins,2), float)
	counted = 0
	for frame in range(startFrame,endFrame,step):
		if frame%100 == 0:
			print ('Frame: ',frame)
		# # Compute normalized bond vectors
		data = node.compute(frame)
		if data.particles.count>1:
			rdf+=data.tables['coordination-rdf'].xy()
			counted +=1
	if counted>0:
		rdf/=counted
		r = rdf[:,0]
		gr = rdf[:,1]
		return r,gr
	else:
		return [0],[0]


def exciID(frame,data):
	"""
	Assigns ID to every particle
	"""

	ID = np.array(range(data.particles.count))
	exci = np.zeros(data.particles.count)
	for  particle in partId:
		index = np.where(partId==particle)
		if (t0[index]-deltat[index]/2)<frame and (t0[index]+deltat[index]/2)>frame:
			exci[particle] = 1
	#print(sum(exci))
	data.particles_.create_property('exci', data=exci)


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






excitations = pd.read_csv('excitation_results_T'+sys.argv[2]+'_tLJ01.csv')
r = np.zeros(numBins)
gr = np.zeros(numBins)
count=0
for keys in ['260']:#excitations:
	if not keys == 'Unnamed: 0':
		node = import_file(sys.argv[1]+'T'+sys.argv[2]+'_N10002_NVT_step_0.1LJ_startFrame'+keys+'.xyz', multiple_frames=True,columns =["Particle Type", "Position.X", "Position.Y", "Position.Z"])
		partId = np.array(excitations[keys][0][1:-1].split(',')).astype(int)
		deltat = np.array(excitations[keys][1][1:-1].split(',')).astype(float)
		t0 = np.array(excitations[keys][2][1:-1].split(',')).astype(float)
		if start:
			r_t,gr_t = gr_calc(node,cut,numBins,excitation=1,startFrame=start,endFrame=end)
		else:
			r_t,gr_t = gr_calc(node,cut,numBins,excitation=1)
		r = r_t
		if len(gr_t)>1:
			count+=1
			gr+=gr_t
		gr=gr/count



#Plot and save the results
pl.rcParams.update({'font.size': 16})
pl.plot(r,gr)
pl.xlabel(r'$r$')
pl.ylabel(r'$g(r)$')
pl.tight_layout()
#pl.savefig('gr_'+sys.argv[2][:-4]+'.pdf')
pl.show()
np.save('gr_'+sys.argv[2][:-4]+'.npy',[r,gr])
