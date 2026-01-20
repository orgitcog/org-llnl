################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Generate ML-HDF for sgcnn and 3dcnn
################################################################################


# basic
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import csv
import h5py
from utils.general_util import log, mkdir_p
import tempfile
from builtins import KeyError
from Bio import PDB
from Bio.PDB import PDBIO
# bio-related
from openbabel import openbabel
from openbabel import pybel
from tfbio.data import Featurizer
from Bio.PDB.PDBParser import PDBParser
import glob
# multi-processing
import multiprocessing as mp
from io import StringIO
import warnings
warnings.filterwarnings("ignore")


class ResSelect(PDB.Select):
	def __init__(self, residueList):
		self.residueList=residueList
	def accept_residue(self, residue):
		if residue in self.residueList:
			return 1
		else:
			return 0

class pdbio(PDBIO):
    def __init__(self, use_model_flag=0, is_pqr=False):
        super(pdbio, self).__init__(use_model_flag, is_pqr)
        self.formatting = _TER_FORMAT_STRING = (
            "TER   %5i      %3s %c%4i%c                                                      \n"
        )

    def save(self, select=ResSelect, write_end=True, preserve_atom_numbering=False):
        """Save structure to a file.

        :param file: output file
        :type file: string or filehandle

        :param select: selects which entities will be written.
        :type select: object

        Typically select is a subclass of L{Select}, it should
        have the following methods:

         - accept_model(model)
         - accept_chain(chain)
         - accept_residue(residue)
         - accept_atom(atom)

        These methods should return 1 if the entity is to be
        written out, 0 otherwise.

        Typically select is a subclass of L{Select}.
        """
        out = ""

        get_atom_line = self._get_atom_line

        # multiple models?
        if len(self.structure) > 1 or self.use_model_flag:
            model_flag = 1
        else:
            model_flag = 0

        for model in self.structure.get_list():
            if not select.accept_model(model):
                continue
            # necessary for ENDMDL
            # do not write ENDMDL if no residues were written
            # for this model
            model_residues_written = 0
            if not preserve_atom_numbering:
                atom_number = 1
            if model_flag:
                out+= f"MODEL      {model.serial_num}\n"

            for chain in model.get_list():
                if not select.accept_chain(chain):
                    continue
                chain_id = chain.id
                if len(chain_id) > 1:
                    raise RuntimeError(
                        f"Chain id ('{chain_id}') exceeds PDB format limit."
                    )

                # necessary for TER
                # do not write TER if no residues were written
                # for this chain
                chain_residues_written = 0

                for residue in chain.get_unpacked_list():
                    if not select.accept_residue(residue):
                        continue
                    hetfield, resseq, icode = residue.id
                    resname = residue.resname
                    segid = residue.segid
                    resid = residue.id[1]
                    if resid > 9999:
                        raise RuntimeError(
                            f"Residue number ('{resid}') exceeds PDB format limit."
                        )

                    for atom in residue.get_unpacked_list():
                        if not select.accept_atom(atom):
                            continue
                        chain_residues_written = 1
                        model_residues_written = 1
                        if preserve_atom_numbering:
                            atom_number = atom.serial_number

                        try:
                            s = get_atom_line(
                                atom,
                                hetfield,
                                segid,
                                atom_number,
                                resname,
                                resseq,
                                icode,
                                chain_id,
                            )
                        except Exception as err:
                            # catch and re-raise with more information
                            raise RuntimeError(
                                f"Error when writing atom {atom.full_id}: {err}"
                            ) from err
                        else:
                            out+=s
                            # inconsequential if preserve_atom_numbering is True
                            atom_number += 1

                if chain_residues_written:
                    out += self.formatting % \
                    (atom_number, resname, chain_id, resseq, icode)

            if model_flag and model_residues_written:
                out+="ENDMDL\n"
        if write_end:
            out+="END   \n"
        
        return out





class CreateHDF5(object):
    r"""
        Generates the hdf5 files with train, val and test splits.
        Args:
            input_dir (str): Directory where the diff dock data is present
            output_dir (str): Directory where to store the processed data
            csv_curate (str): Directory to check if a train, val and test split is present
            dataset (str): Name of the dataset to be created
            pocket (str): specific pocket (if available)
    """
    def __init__(
            self,
            input_dir="",
            output_dir="", 
            csv_curate="",
            dataset="dengue",
            pocket="",
            pdb_dir="",
            radius=8
        ):
        
        assert os.path.isdir(input_dir), "Directory {} does not exist. Aborting...".format(input_dir)
        assert len(os.listdir(input_dir)) > 1, "Directory is empty. Aborting..."
        assert os.path.isfile("elements.xml"), ValueError("elements.xml does not exist. Aborting...")
        

        dirs = os.listdir(input_dir)
        dir_inds = [int(dir.strip('dock_proc').split('.')[0]) for dir in dirs]
        log.infov("Directory contains dock_proc{}.hdf5 to dock_proc{}.hdf5".format(min(dir_inds), max(dir_inds)))
        dirs = [input_dir + dir for dir in dirs if os.path.isfile(input_dir+dir)]
        self.dataset = dataset
        element_info_dict = dict()
        element_info_xml = ET.parse("elements.xml")
        for element in element_info_xml.iter():
            if "comment" in element.attrib.keys():
                continue
            else:
                element_info_dict[int(element.attrib["number"])] = element.attrib
        self.element_info_dict = element_info_dict

        
        tempfile.tempdir = mkdir_p(output_dir)
        
        self.output_dir = output_dir
        log.infov("Data directory: {}".format(output_dir))

        self.input_dir = input_dir
        
        self.input_dir_list = sorted(os.listdir(input_dir))

        self.pdbdict = self.load_anno_csv(csv_path=csv_curate)

        self.pocket = pocket
        self.receptor_path = os.path.join(
            os.path.abspath(os.path.join(self.input_dir, "..")), 
            "receptor.hdf5"
        )
        p = PDBParser()
        if pdb_dir:
            self.receptor_pdb = p.get_structure('rec', pdb_dir)

        self.OBConverter = openbabel.OBConversion()

        self.compute_partial_charge = True
        self.radius = float(radius)

    def load_anno_csv(
            self,
            csv_path, 
            pdbid_ind=0, 
            label_ind=4
        ):
        pdb_dict = {}

        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                pdbid = row[pdbid_ind]
                label = float(row[label_ind])
                pdb_dict[pdbid] = (pdbid, label)
                
        return pdb_dict
    
    def featurize_pybel_complex(
            self,
            ligand_mol, 
            pocket_mol, 
        ) -> np.array:

        featurizer = Featurizer()

        # get ligand features
        ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1)

        # get processed pocket features
        pocket_coords, pocket_features = featurizer.get_features(pocket_mol, molcode=-1)

        # center the coordinates on the ligand coordinates
        centroid_ligand = ligand_coords.mean(axis=0)
        ligand_coords -= centroid_ligand

        pocket_coords -= centroid_ligand
        data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)), np.concatenate((ligand_features, pocket_features))), axis=1)

        return data


    def extractPDBPoses(self, pdbqt_data) -> dict:
        pdb_pose_list = []
        pdb_pose_str = ""
        for line in pdbqt_data:
            if line.startswith("MODEL"):
                pdb_pose_str = ""
                continue

            # Do not write ROOT, ENDROOT, BRANCH, ENDBRANCH, TORSDOF records.
            if line.startswith ('ROOT') or line.startswith ('ENDROOT') \
                or line.startswith ('BRANCH') or line.startswith ('ENDBRANCH') \
                or line.startswith ('TORSDOF'):
                continue

            pdb_pose_str += line.rstrip() + '\n'
            if line.startswith("ENDMDL"):
                pdb_pose_list.append(pdb_pose_str)
        
        return pdb_pose_list
    
    def cutProtein(self, lig_pdb_fn):
        p = PDBParser()
        lig = p.get_structure('lig', StringIO(lig_pdb_fn))
        d2=float(self.radius)**2
        residueList=[]
        for residue in self.receptor_pdb.get_residues():
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
            return [], False
        
        io = pdbio(use_model_flag=True)
        io.set_structure(self.receptor_pdb)
        data = io.save(select=ResSelect(residueList))
        # save the file to a pdb file
        with open(os.path.join(self.output_dir,'post_surgery_receptor.pdb'), 'w') as file:
            file.write(data)
        return data, True
    
    def get_mol_from_pdb(self, pdb_pose_str, compute_partial_charge=False):

        if compute_partial_charge == True:
            # need to convert pdbqt -> mol2 -> pdbqt
            self.OBConverter.SetInAndOutFormats("pdbqt", "mol2")
            mol = openbabel.OBMol()
            self.OBConverter.ReadString(mol, pdb_pose_str)
            mol2_pose_str = self.OBConverter.WriteString(mol)

            self.OBConverter.SetInAndOutFormats("mol2", "pdbqt")
            mol = openbabel.OBMol()
            self.OBConverter.ReadString(mol, mol2_pose_str)
            pdb_pose_str_pc = self.OBConverter.WriteString(mol)
            pdb_pose_str = pdb_pose_str_pc

        # note that pdbqt -> mol2 make some feature changes
        self.OBConverter.SetInAndOutFormats("pdbqt", "mol2")
        self.OBConverter.ReadString(mol, pdb_pose_str)
        mol2_pose_str = self.OBConverter.WriteString(mol)
        return pybel.readstring('mol2', mol2_pose_str)
    


    def dumpHDF5(self) -> None:
        # Creating openbabel version
        receptor_hd = h5py.File(self.receptor_path, 'r')
        receptor_list = receptor_hd["rec"]
        
        log.warning("Saved to {}".format(os.path.join(self.output_dir, self.dataset+"-"+self.pocket+".hdf")))
        with h5py.File(os.path.join(self.output_dir, self.dataset+"-"+self.pocket+".hdf"), 'w') as trainFile:
            for dock_proc in self.input_dir_list:
                ligand_hd = h5py.File(os.path.join(self.input_dir, dock_proc), 'r')
                ligand_list = ligand_hd["dock"]
                log.infov("Working on directory {}".format(os.path.join(self.input_dir, dock_proc)))
                for key in ligand_list:

                    assert key in receptor_list, KeyError("Make sure the receptor key and the ligand match")
                    for ligand in ligand_list[key]:
                        if not ligand in self.pdbdict:
                            continue
                        label = self.pdbdict[ligand][1]

                        rec_pdbqt = (
                            receptor_list[key]["file"]["rec_min.pdbqt"][()]
                            .tobytes()
                            .decode('utf-8')
                        ).rstrip('\x00')
                        
                        if not ligand_list[key][ligand]["meta"]["numPose"][0]:
                            log_msg = "[%s - %s] pose count: %d, pose scores: NO POSES!" %(
                                    dock_proc, 
                                    ligand, 
                                    ligand_list[key][ligand]["meta"]["numPose"][0],
                                )
                            log.error(log_msg)

                            with open(os.path.join(self.output_dir, "error_log.txt"), "a") as f:
                                f.write(log_msg + "\n")
                            continue
                        
                        pdbqt = ligand_list[key][ligand]["file"]["poses.pdbqt"][()]
                        pdbqt = pdbqt.tobytes().decode('utf-8')
                        temp, flag = self.cutProtein(pdbqt.rstrip("\x00"))
                        if flag:
                            rec_pdbqt = temp
                        
                        rec_mol = self.get_mol_from_pdb(rec_pdbqt, self.compute_partial_charge)
                        rec_vdw = [
                            float(self.element_info_dict[atom.atomicnum]["vdWRadius"]) \
                                for atom in rec_mol.atoms \
                                if not atom.atomicnum in (0,1)
                        ]
                        
                        log_msg = "[%s - %s] pose count: %d, pose scores: %s"%(
                                dock_proc, 
                                ligand, 
                                ligand_list[key][ligand]["meta"]["numPose"][0],
                                list(
                                    ligand_list[key][ligand]["meta"]["scores"][pose][0] \
                                    for pose in ligand_list[key][ligand]["meta"]["scores"]
                                )
                            )
                        log.warning(log_msg)
                        with open(os.path.join(self.output_dir, "success_log.txt"), "a") as f:
                            f.write(log_msg + "\n")

                        pose_list = self.extractPDBPoses(pdbqt.split('\n'))

                        grp = trainFile.create_group(ligand)
                        grp.attrs["affinity"] = label

                        pybel_grp = grp.create_group("pybel")
                        processed_grp = pybel_grp.create_group("processed")
                        docking_grp = processed_grp.create_group("docking")

                        poses = list(ligand_list[key][ligand]["meta"]["scores"])
                        for index in poses:
                            lig_pose_pdb = pose_list[int(index)-1]
                            lig_mol = self.get_mol_from_pdb(lig_pose_pdb, self.compute_partial_charge)

                            comp_data = self.featurize_pybel_complex(
                                ligand_mol=lig_mol, 
                                pocket_mol=rec_mol
                            )
                            lig_vdw = [float(self.element_info_dict[atom.atomicnum]["vdWRadius"]) \
                                for atom in lig_mol.atoms \
                                if not atom.atomicnum in (0,1)
                            ]

                            comp_vdw = np.concatenate(
                                [np.asarray(lig_vdw).reshape(-1), 
                                    np.asarray(rec_vdw).reshape(-1)], 
                                axis=0
                            )
                            assert comp_vdw.shape[0] == comp_data.shape[0]
                            pose_grp = docking_grp.create_group(str(index))
                            pose_grp.attrs["van_der_waals"] = comp_vdw
                            pose_grp.attrs["score"] = ligand_list[key][ligand]["meta"]["scores"][index][0]
                            

                            pose_dataset = pose_grp.create_dataset("data", data=comp_data, shape=comp_data.shape, dtype='float32', compression='lzf')
        
        return pose_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str,
        default="/p/vast1/bcwc/flavivirus/data/flavivirus_docking_results/"
    )
    parser.add_argument(
        "-d","--dataset", type=str, required=False,
        default="dengue", help="Path to receptor file"
    )
    parser.add_argument(
        "-s","--sub-dataset",
        default="denv2", 
        help="Path to docking files"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=""
    )
    parser.add_argument(
        "--pocket-type", type=str,
        default="2fom"
    )
    args = parser.parse_args()


    ligand_dir = os.path.join(args.dir + "{}/{}".format(
            args.dataset, 
            args.sub_dataset
        ), 
        "{}/scratch/dockHDF5/".format(
            args.pocket_type
        )
    )

    pdbfile = glob.glob("*.pdb", root_dir=os.path.join(args.dir+"{}/{}/{}".format(
                args.dataset, 
                args.sub_dataset,
                args.pocket_type
            )
        )
    )

    pdb_dir = os.path.join(args.dir+"{}/{}/{}/".format(
            args.dataset, 
            args.sub_dataset,
            args.pocket_type
        ), 
        pdbfile[0]
    )
    output_dir = os.path.join(
        "data/{}".format(args.dataset),
        "{}".format(args.sub_dataset)
    )
    
    obj = CreateHDF5(
        input_dir=ligand_dir, 
        output_dir=output_dir, 
        csv_curate=args.dir + "protease_ligand_prep.csv",
        pocket=args.pocket_type,
        dataset=args.dataset,
        pdb_dir=pdb_dir
    )
    obj.dumpHDF5()