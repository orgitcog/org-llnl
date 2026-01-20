from trata.kosh_sampler import KoshSampler
import kosh
import h5py
import numpy as np
from sklearn import datasets
import os


def test_KoshSampler():

    import trata.sampler as sampler
    from sklearn.gaussian_process import GaussianProcessRegressor
    import string
    import random

    rand_n = 7

    test_box = [[-5, 5], [-5, 5]]
    starting_data = sampler.LatinHyperCubeSampler.sample_points(box=test_box,
                                                                num_points=25,
                                                                seed=2018)
    starting_output = starting_data[:, 0] * starting_data[:, 1] + 50.0

    gpm = GaussianProcessRegressor()
    model = gpm.fit(starting_data, starting_output)

    # generate random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                                 string.digits, k=rand_n))
    fileName = 'data_' + str(res) + '.h5'

    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('inputs', data=starting_data.astype(np.float64))
    h5f.create_dataset('outputs', data=starting_output.astype(np.float64))
    h5f.close()

    # Create a new store (erase if exists)
    store_name = str(res) + 'store.sql'
    store = kosh.connect(store_name, delete_all_contents=True)
    dataset = store.create("kosh_example1")
    dataset.associate([fileName], "hdf5")

    num_points = 7
    ndim = starting_data.shape[1]

    new_points = KoshSampler(dataset['inputs'],
                             method='ActiveLearningSampler',
                             num_points=7,
                             model=model,
                             num_cand_points=20,
                             box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    new_points = KoshSampler(dataset['inputs'],
                             method='BestCandidateSampler',
                             num_points=7,
                             num_cand_points=20,
                             box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    new_points = KoshSampler(dataset['inputs'],
                                          method='DeltaSampler',
                                          num_points=7,
                                          model=model,
                                          output=dataset['outputs'],
                                          num_cand_points=20,
                                          box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    new_points = KoshSampler(dataset['inputs'],
                                          method='ExpectedImprovementSampler',
                                          num_points=7,
                                          model=model,
                                          output=dataset['outputs'],
                                          num_cand_points=20,
                                          box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    new_points = KoshSampler(dataset['inputs'],
                                          method='LearningExpectedImprovementSampler',
                                          num_points=7,
                                          model=model,
                                          output=dataset['outputs'],
                                          num_cand_points=20,
                                          box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    # Cleanup
    os.remove(fileName)
    os.remove(store_name)
    store.close()
