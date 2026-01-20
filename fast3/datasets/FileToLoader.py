import os
import numpy as np
import pandas as pd
import h5py
import argparse
import pybel


class FileToLoader(object):
    def __init__(self, csv_path, metadata) -> None:
        self.path = csv_path
        self.metadata = metadata
        # TODO
        # Check if .hdf file exists
        if not os.isdir(self.path):
            # If not, check if raw files exist
            pass
                # if not, check if the tar file exist
                    # if not, call download
                # else
                    # call load
        
        # If not, download and create it
        self.affinity_data = pd.read_csv(metadata)
        


    def download(self) -> None:
        # TODO: Check if the file exists
        # If not, download is required
        pass

    def load(self) -> None:
        # TODO:
        # Load the data from self.path

        pass

    def dump(self) -> None:
        # TODO:
        # Dump all the data in the folder
        pass

    def unzip(self) -> list:
        pass

    def parse_mol_vdw(self) -> np.array:
        pass

    def featurize_pybel_complex(self) -> np.array:
        pass



    