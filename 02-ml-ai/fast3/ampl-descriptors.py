# importing relevant libraries
import pandas as pd
pd.set_option('display.max_columns', None)
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse

# Set up
dataset_file = 'data/pdbbind2020/pdbbind_v2020_converted_train_valid_test_scaffold_c35aeaab-910c-4dcf-8f9f-04b55179aa1a.csv'
odir='dataset/SLC6A3_models/'

response_col = "affinity"
compound_id = "pdbid"
smiles_col = "base_rdkit_smiles"
split_uuid = "c35aeaab-910c-4dcf-8f9f-04b55179aa1a"

params = {
        "prediction_type": "regression",
        "dataset_key": dataset_file,
        "id_col": compound_id,
        "smiles_col": smiles_col,
        "response_cols": response_col,
        "previously_split": "True",
        "split_uuid" : split_uuid,
        "split_only": "False",
        "featurizer": "computed_descriptors",
        "descriptor_type" : "rdkit_raw",
        "model_type": "RF",
        "verbose": "True",
        "transformers": "True",
        "rerun": "False",
        "result_dir": odir
    }

ampl_param = parse.wrapper(params)
pl = mp.ModelPipeline(ampl_param)
pl.train_model()
