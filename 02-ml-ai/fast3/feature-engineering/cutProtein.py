# original implementation by X. Zhang
# modified by Jonathan Allen, Hyojin Kim
# 2021/06/09

import os
import h5py
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from RawToHDF5 import pdbio, ResSelect
from io import StringIO
		

class cutProtein(object):
	def __init__(
			self, 
			ligand_dir: str,
			receptor_dir: str
	):
		self.ligand_dir = ligand_dir
		self.receptor_dir = receptor_dir

	def ProteinSurgery(self):
		p = PDB.PDBParser()
		# lig = p.get_structure('ligand', lig_pdb_fn)

def cutProtein(lig_pdb_fn, rec_pdb_fn, radius=float('inf')):
	p = PDB.PDBParser()
	lig = h5py.File(lig_pdb_fn, 'r')
	lig = lig["dock"]["2fom_moe_prep"]["2800"]["file"]["poses.pdbqt"][()][()] \
							.tobytes() \
                            .decode('utf-8')\
                        	.rstrip('\x00')
	lig = StringIO(lig)
	lig = p.get_structure('lig', lig)
	rec = p.get_structure('rec', rec_pdb_fn)
	d=float(radius)
	d2=d*d
	residueList=[]
	for residue in rec.get_residues():
		skipRes=False
		for atomI in residue:
			[xi, yi, zi]=atomI.get_coord()
			for atomJ in lig.get_atoms():
				[xj, yj, zj]=atomJ.get_coord()
				dx=xi-xj
				dy=yi-yj
				dz=zi-zj
				dist=dx*dx+dy*dy+dz*dz
				if dist<d2:
					residueList.append(residue)
					skipRes = True
					break
			if skipRes:
				break

	if len(residueList) == 0:
		return False
	from pdb import set_trace
	
	io = pdbio()
	io.set_structure(rec)
	dat = io.save(select=ResSelect(residueList))
	set_trace()
	
	return True

def extract_sequence(rec_pdb_fn):
	d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', \
	  	'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', \
		'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', \
		'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
	
	parser = PDBParser(QUIET=True)
	structure = parser.get_structure('struct', rec_pdb_fn)

	seq_str = ""
	for model in structure:
		for chain in model:
			seq = []
			for residue in chain:
				seq.append(d3to1[residue.resname])
			seq_str = ''.join(seq)
			print(seq_str)

	print("final: ", seq_str)
	return seq_str



def get_subdirectories(dir):
	file_list = os.listdir(dir)
	return [dir for dir in file_list if os.path.isdir(dir)]


# temp
import sys
sys.stdout.flush()

rec = "/p/vast1/bcwc/flavivirus/data/flavivirus_docking_results/dengue/denv2/2fom/2fom_moe_prep.pdb"
lig = "/p/vast1/bcwc/flavivirus/data/flavivirus_docking_results/dengue/denv2/2fom/scratch/dockHDF5/dock_proc1.hdf5"
cutProtein(lig_pdb_fn=lig, 
		   rec_pdb_fn=rec,)
# Get the pdbqt data from the receptor
# Then get the pdbqt data for the ligand from the hdf5 file
