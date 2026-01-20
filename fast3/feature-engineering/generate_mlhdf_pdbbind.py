# Generate ML-HDF as ConveyorLC pipeline
# For receptor-ligand data conversion from pdbbind dataset
# Hyojin Kim
# 2020/10/21


# basic
import os
import subprocess
import argparse
import warnings
import numpy as np
import xml.etree.ElementTree as ET
import csv
import h5py
import pandas as pd
from pdb import set_trace

# bio-related
from openbabel import openbabel
from openbabel import pybel
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
from tfbio.data import Featurizer


# decide whether extracting feature from pdb/pdbqt or from mol2
#g_convert_mol2 = True # always need to convert to mol2 first because of feature difference between pdbqt and mol2
g_compute_partial_charge = True


# set up program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--receptor-path", default="", help="input receptor hdf5 filepath")
parser.add_argument("--ligand-dir", default="", help="input ligand hdf5 dir")
parser.add_argument("--csv-path", default="pdbbind_2019_with_measurement_annotation.csv", help="csv annotation file path")
parser.add_argument("--output-dataset", default="casf-2016", help="output dataset")
parser.add_argument("--output-path", default="", help="output ml-hdf file path")
args = parser.parse_args()


##### temp start #####
#args.receptor_path = "/Users/kim63/Desktop/temp_pdbbind2019/receptor.hdf5"
#args.ligand_dir = "/Users/kim63/Desktop/temp_pdbbind2019/dockHDF5"
#args.csv_path = "pdbbind_2019_with_measurement_annotation.csv"
#args.output_dataset = "casf-2016"
#args.output_dataset = "refined"
#args.output_path = "/Users/kim63/Desktop/temp_pdbbind2019/pdbbind2019_casf-2016.hdf"
##### temp end #####



def load_anno_csv(csv_path, pdbid_ind=0, label_ind=4):
    pdb_dict = {}

    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            pdbid = row[pdbid_ind]
            label = float(row[label_ind])
            pdb_dict[pdbid] = (pdbid, label)
			
    return pdb_dict


def extract_pdb_poses(pdbqt_data):
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


def get_mol_from_pdb(pdb_pose_str, compute_partial_charge=False):
    ob_conversion = openbabel.OBConversion()

    if compute_partial_charge == True:
        # need to convert pdbqt -> mol2 -> pdbqt
        ob_conversion.SetInAndOutFormats("pdbqt", "mol2")
        mol = openbabel.OBMol()
        ob_conversion.ReadString(mol, pdb_pose_str)
        mol2_pose_str = ob_conversion.WriteString(mol)

        ob_conversion.SetInAndOutFormats("mol2", "pdbqt")
        mol = openbabel.OBMol()
        ob_conversion.ReadString(mol, mol2_pose_str)
        pdb_pose_str_pc = ob_conversion.WriteString(mol)
        pdb_pose_str = pdb_pose_str_pc

    # note that pdbqt -> mol2 make some feature changes
    ob_conversion.SetInAndOutFormats("pdbqt", "mol2")
    ob_conversion.ReadString(mol, pdb_pose_str)
    mol2_pose_str = ob_conversion.WriteString(mol)
    return pybel.readstring('mol2', mol2_pose_str)
    #    mol = openbabel.OBMol()
    #    ob_conversion.ReadString(mol, pdb_pose_str)
    #    mol2_pose_str = ob_conversion.WriteString(mol)
    #    return pybel.readstring('mol2', mol2_pose_str)
    #else:
    #    return pybel.readstring('pdbqt', pdb_pose_str)


def read_element_desc(desc_file):
    element_info_dict = {}
    element_info_xml = ET.parse(desc_file)
    for element in element_info_xml.iter():
        if "comment" in element.attrib.keys():
            continue
        else:
            element_info_dict[int(element.attrib["number"])] = element.attrib

    return element_info_dict


def parse_mol_vdw(mol, element_dict):
    vdw_list = []

    if isinstance(mol, pybel.Molecule):
        for atom in mol.atoms:
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.atomicnum) == 1:
                continue
            if int(atom.atomicnum) == 0:
                continue
            else:
                vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))

    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        for atom in mol.GetAtoms():
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.GetAtomicNum()) == 1:
                continue
            else:
                vdw_list.append(float(element_dict[atom.GetAtomicNum()]["vdWRadius"]))
    else:
        raise RuntimeError("must provide a pybel mol or an RDKIT mol")

    return np.asarray(vdw_list)


def featurize_pybel_complex(ligand_mol, pocket_mol, name):

    featurizer = Featurizer()
    
    #for feat_name in featurizer.FEATURE_NAMES:
    #	feat_ind = featurizer.FEATURE_NAMES.index(feat_name)
    #	print(feat_name, feat_ind)
    	
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

    # get ligand features
    ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1)

    #if not (ligand_features[:, charge_idx] != 0).any():  # ensures that partial charge on all atoms is non-zero?
    #    raise RuntimeError("invalid charges for the ligand {}".format(name))

    # get processed pocket features
    pocket_coords, pocket_features = featurizer.get_features(pocket_mol, molcode=-1)
    #if not (pocket_features[:, charge_idx] != 0).any():
    #    raise RuntimeError("invalid charges for the pocket {}".format(name))

    # center the coordinates on the ligand coordinates
    centroid_ligand = ligand_coords.mean(axis=0)
    ligand_coords -= centroid_ligand

    pocket_coords -= centroid_ligand
    data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)), np.concatenate((ligand_features, pocket_features))), axis=1)

    return data


def get_files_ext(a_dir, a_ext):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name)) and name.endswith(a_ext)]


def valid_file(a_path):
    return os.path.isfile(a_path) and os.path.getsize(a_path) > 0


def main():
    # load annotation csv file
    pdb_dict = load_anno_csv(args.csv_path)

    # initialize element dict
    element_dict = read_element_desc("elements.xml")

    # initialize failure dict
    failure_dict = {"name": [], "partition": [], "set": [], "error": []}

    # initialize docking score list
    docking_score_dict = {"ligand_file": [], "pdbid": [], "poseid": [], "VINA": []}

	# get ligand file list
    ligand_file_list = get_files_ext(args.ligand_dir, "hdf5")
    ligand_file_list.sort()
    output_docking_score_fn = "docking_scores.csv"

    # open receptor hdf
    receptor_hd = h5py.File(args.receptor_path, 'r')
    receptor_list = receptor_hd["rec"]

    # create output ml-hdf
    with h5py.File(args.output_path, 'w') as output_ml_hdf:

        # loop over working_file_list
        for ligand_fn in ligand_file_list:

            # open ligand hdf
            ligand_hd = h5py.File(os.path.join(args.ligand_dir, ligand_fn), 'r')
            ligand_list = ligand_hd["dock"]

            # get ligand id list
            key_list = list(ligand_list.keys())

            # for each ligand
            for key_name in key_list:
                line_ids = list(ligand_list[key_name])
                for pdbid in line_ids:

                    if not pdbid in pdb_dict:
                        continue
                    
                    label = pdb_dict[pdbid][1]
                    # set = pdb_dict[pdbid][2]
                    # if set != args.output_dataset:
                    #     continue
                    print(pdbid)

                    # get receptor pdb
                    rec_pdbqt = receptor_list[key_name]["file"]["rec_min.pdbqt"][()]
                    rec_pdbqt = rec_pdbqt.tobytes().decode('utf-8')
                    if rec_pdbqt[-1] == '\0':
                        rec_pdbqt = rec_pdbqt.rstrip('\x00')
                    rec_mol = get_mol_from_pdb(rec_pdbqt, g_compute_partial_charge)
                    rec_vdw = parse_mol_vdw(mol=rec_mol, element_dict=element_dict)
                    #print(len(rec_mol.atoms))
                    
                    # get ligand info (score, etc)
                    lig_id = str(list(ligand_list[key_name].keys())[0])
                    pose_count = ligand_list[key_name][lig_id]["meta"]["numPose"][0]
                    set_trace()
                    pose_scores = [ligand_list[key_name][lig_id]["meta"]["scores"][str(pose_id+1)][0] for pose_id in range(pose_count)]
                    print("[%s - %s] pose count: %d, pose scores: %s" % (ligand_fn, pdbid, pose_count, str(pose_scores)))
                    set_trace()
                    # add docking scores to list
                    for pose_ind in range(pose_count):
                        docking_score_dict["ligand_file"].append(ligand_fn)
                        docking_score_dict["pdbid"].append(pdbid)
                        docking_score_dict["poseid"].append(pose_ind+1)
                        docking_score_dict["VINA"].append(pose_scores[pose_ind])

                    # get ligand pdb
                    lig_pdbqt = ligand_list[key_name][lig_id]["file"]["poses.pdbqt"][()]
                    lig_pdbqt = lig_pdbqt.tobytes().decode('utf-8')
                    lig_pdb_pose_list = extract_pdb_poses(lig_pdbqt.split('\n'))

                    # create output pdb group (a complex with multiple poses)
                    grp = output_ml_hdf.create_group(pdbid)
                    grp.attrs["affinity"] = label
    ####                grp.attrs["ligand features "] = XXXXX
                    pybel_grp = grp.create_group("pybel")
                    processed_grp = pybel_grp.create_group("processed")
                    docking_grp = processed_grp.create_group("docking")

                    # for each ligand pose
                    for pose_ind in range(pose_count):
                        lig_pose_pdb = lig_pdb_pose_list[pose_ind]

                        # get ligand mol instance from pdb pose data (string)
                        lig_pose_mol = get_mol_from_pdb(lig_pose_pdb, g_compute_partial_charge)
                        #print(len(lig_pose_mol.atoms))

                        # extract feature
                        comp_data = featurize_pybel_complex(ligand_mol=lig_pose_mol, pocket_mol=rec_mol, name="%s_%s" % (pdbid, pose_ind+1))

                        # extract the van der waals radii for the ligand/pocket
                        lig_vdw = parse_mol_vdw(mol=lig_pose_mol, element_dict=element_dict)
                        comp_vdw = np.concatenate([lig_vdw.reshape(-1), rec_vdw.reshape(-1)], axis=0)
                        assert comp_vdw.shape[0] == comp_data.shape[0]

                        # read additional feature from ligand/receptor/complex
                        # ligand_hd["dock"][...]

                        # create output pdb pose group
                        pose_grp = docking_grp.create_group(str(pose_ind+1))
                        pose_grp.attrs["van_der_waals"] = comp_vdw

    ####                    pose_grp.attrs["interaction features"] = aaaa
                        
                        pose_dataset = pose_grp.create_dataset("data", data=comp_data, shape=comp_data.shape, dtype='float32', compression='lzf')


if __name__ == "__main__":
    main()