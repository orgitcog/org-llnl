import argparse
import itertools
import math
import time
import os
import pandas as pd
import numpy as np
import code_from_authors as cfa

def parse_args():
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--csv', 
        help='Path to input CSV containing proteins and ligands. \
            Must be comma separated.')
    parser.add_argument('--smiles_col',
            help='The column in the csv file that contains \
                SMILES strings.')
    parser.add_argument('--protein_col',
            help='The column in the csv file that contains \
                proteins.')
    parser.add_argument('--num_workers',
            default=1,
            type=int,
            help='Number of workers for parallel feature calc.')

    # output parameters
    parser.add_argument('--ecfp_csv', 
            default='',
            help='Output containing ECFP features.\
                One ECFP feature per column, colums \
                follow ecfp_0, ecfp_1, ecfp_2 pattern')
    parser.add_argument('--psc_csv', 
            default='',
            help='Output containing PSC features.\
                One PSC feature per column, colums \
                follow psc_0, psc_1, psc_2 pattern')
    
    args = parser.parse_args()
    if args.ecfp_csv == '':
        args.ecfp_csv = f"{os.path.splitext(args.csv)[0]}_ECFP.csv"

    if args.psc_csv == '':
        args.psc_csv = f"{os.path.splitext(args.csv)[0]}_PSC.csv"

    return args

def parallel(things, single_function, workers):
    """Parallelize a function

    Breaks up things and feeds them into single_function. Basically just
    uses map, but handles batching for you.

    Paramters
    ---------
    things: List[objs]
        A list of things to get batched up

    single_function: function(List[objs])
        A function that takes a list of things as inputs and returns 
        a list of strings and numpy arrays
        Rows matches len(things)

    workers: int
        Number of parallel workers

    Returns
    -------
        A list of outputs from single_function
    """
    from functools import partial
    func = partial(parallel, single_function=single_function, workers=1)
    if workers > 1:
      from multiprocessing import pool
      batchsize = math.ceil(len(things)/workers)
      batches = [things[i:i+batchsize] for i in range(0, len(things), batchsize)]
      with pool.Pool(workers) as p:
        outputs = p.map(func,batches)
        strout = [o[0] for o in outputs]
        outs = [o[1] for o in outputs]
        result = np.vstack(outs)
        strout = list(itertools.chain.from_iterable(strout))
    else:
        strout, result = single_function(things)

    return strout, result

if __name__ == '__main__':
    # Example command line: python generate.py --csv ../DrugBAN/datasets/bindingdb/full.csv --smiles_col SMILES --protein_col Protein --num_workers 8
    # This will create ../DrugBAN/datasets/bindingdb/full_ECFP.csv and ../DrugBAN/datasets/bindingdb/full_PSC.csv
    # get the start time
    start_time = time.time()

    args = parse_args()
    data_dir = "data/dengue/denv2"
    comp_infos = pd.read_csv(os.path.join(data_dir, "protease_ligand_prep.csv"))
    comp_ids = pd.read_csv(os.path.join(data_dir, "train_test_valid_ids_scaffold.csv"))
    comp_ids = comp_ids[comp_ids["subset"]=="train"]["cmpd_id"].tolist()
    comp_infos = comp_infos[comp_infos["compound_id"].isin(comp_ids)]
    smiles = comp_infos[args.smiles_col].unique().tolist()

    

    print(comp_infos.columns)

    proteins = comp_infos[args.protein_col].unique().tolist()
    
    print('There are ', len(proteins), 'unique proteins.')

    good_proteins, psc_features = parallel(proteins, single_function=cfa.get_3mer_encoding, workers=args.num_workers)

    psc_col_names = [f'psc_{i}' for i in range(psc_features.shape[1])]
    psc_df = pd.DataFrame(psc_features, columns=psc_col_names)
    psc_df[args.protein_col] = good_proteins

    print('Saving PSC features to', args.psc_csv)
    psc_df.to_csv(args.psc_csv, index=False)

    print('There are', len(smiles), 'unique ligands.')

    good_smiles, ecfp_features = parallel(smiles, single_function=cfa.get_ecfp_encoding, workers=args.num_workers)
    ecfp_col_names = [f'ecfp_{i}' for i in range(ecfp_features.shape[1])]

    ecfp_df = pd.DataFrame(ecfp_features, columns=ecfp_col_names)
    ecfp_df[args.smiles_col] = good_smiles
    
    print('Saving ECFP features to', args.ecfp_csv)
    ecfp_df.to_csv(args.ecfp_csv, index=False)

    # get the end time
    end_time = time.time()

    # get the execution time
    elapsed_time = end_time-start_time
    print('Execution time:', elapsed_time, 'seconds')
