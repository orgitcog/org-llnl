from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pytest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.datasets import make_blobs
import trata.sampler
import trata.adaptive_sampler


def helper_function(np_input):
    np_input = np_input.astype(float)
    out = np.sin(np_input[:, 0]) + np.cos(np_input[:, 1])
    return out.reshape(-1, 1)


ls_test_box = [[-7.0, 9.0], [-2.0, 10.0]]
np_train_input, y = make_blobs(n_samples=50, centers=3, n_features=2, random_state=3)

np_train_output = helper_function(np_train_input)
surrogate_model = gpr().fit(np_train_input, np_train_output)
np_candidate_points = trata.sampler.LatinHyperCubeSampler.sample_points(num_points=50,
                                                                        box=ls_test_box,
                                                                        seed=2019)


def test_ActiveLearningSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        ActiveLearningSampler.sample_points(num_points=5,
                                            cand_points=np_candidate_points,
                                            model=surrogate_model)
    np_expected_values = [[-6.983927383565486, 7.2401358376235265],
                          [-3.9445248702879296, 9.70869629305285],
                          [-5.305982276357915, 8.239955581495817],
                          [7.798240204471682, 0.3501770869725567],
                          [-4.165945669390727, 9.93303611553577]]

    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)


def test_ActiveLearningSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ActiveLearningSampler.sample_points,
                  cand_points=np_candidate_points,
                  model=surrogate_model)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ActiveLearningSampler.sample_points,
                  num_points=5,
                  model=surrogate_model)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.ActiveLearningSampler.sample_points,
                  num_points=5,
                  num_cand_points=np_candidate_points)


def test_BestCandidateSampler_valid():
    # Testing with previously generated candidate points
    np_actual_values = trata.adaptive_sampler. \
      BestCandidateSampler.sample_points(num_points=5,
                                         values=np_train_input,
                                         cand_points=np_candidate_points)
    np_expected_values = np.array([[-6.983927383565486, 7.2401358376235265],
                                   [-5.305982276357915, 8.239955581495817],
                                   [7.798240204471682, 0.3501770869725567],
                                   [-3.9445248702879296, 9.70869629305285],
                                   [-4.165945669390727, 9.93303611553577]])
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)
    # Testing with generating candidate points
    np_actual_values = trata.adaptive_sampler. \
      BestCandidateSampler.sample_points(num_points=5,
                                         num_cand_points=50,
                                         box=ls_test_box,
                                         values=np_train_input,
                                         seed=2019)
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)


def test_BestCandidateSampler_invalid():
    # Number of ranges in box and number of features don't match
    pytest.raises(ValueError, trata.adaptive_sampler.BestCandidateSampler.sample_points,
                  num_points=5,
                  values=np_train_input,
                  num_cand_points=50,
                  box=[[-7.0, 9.0], [-2.0, 10.0], [3.5, 7.8]])


def test_DeltaSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        DeltaSampler.sample_points(num_points=5,
                                   cand_points=np_candidate_points,
                                   model=surrogate_model,
                                   values=np_train_input,
                                   output=np_train_output)
    np_expected_values = [[6.713400984317753, 3.01647745081449],
                          [7.798240204471682, 0.3501770869725567],
                          [1.1646502174706903, 9.06599351329055],
                          [0.2925519459646324, 8.854669731054763],
                          [-0.7792055024578879, 8.558647821903405]]

    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)


def test_DeltaSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  values=np_train_input,
                  output=np_train_output)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                  num_points=5,
                  model=surrogate_model,
                  values=np_train_input,
                  output=np_train_output)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  values=np_train_input,
                  output=np_train_output)
    # values not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  output=np_train_output)
    # output not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  values=np_train_input)


def test_ExpectedImprovementSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        ExpectedImprovementSampler.sample_points(num_points=5,
                                                 cand_points=np_candidate_points,
                                                 model=surrogate_model,
                                                 values=np_train_input,
                                                 output=np_train_output)
    np_expected_values = [[6.713400984317753, 3.01647745081449],
                          [7.798240204471682, 0.3501770869725567],
                          [1.1646502174706903, 9.06599351329055],
                          [0.2925519459646324, 8.854669731054763],
                          [-0.7792055024578879, 8.558647821903405]]

    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)


def test_ExpectedImprovementSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  values=np_train_input,
                  output=np_train_output)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                  num_points=5,
                  model=surrogate_model,
                  values=np_train_input,
                  output=np_train_output)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  values=np_train_input,
                  output=np_train_output)
    # values not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  output=np_train_output)
    # output not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  values=np_train_input)


def test_LearningExpectedImprovementSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        LearningExpectedImprovementSampler.sample_points(num_points=5,
                                                         cand_points=np_candidate_points,
                                                         model=surrogate_model,
                                                         values=np_train_input,
                                                         output=np_train_output)
    np_expected_values = [[6.713400984317753, 3.01647745081449],
                          [7.798240204471682, 0.3501770869725567],
                          [1.1646502174706903, 9.06599351329055],
                          [0.2925519459646324, 8.854669731054763],
                          [-0.7792055024578879, 8.558647821903405]]

    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)


def test_LearningExpectedImprovementSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  values=np_train_input,
                  output=np_train_output)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                  num_points=5,
                  model=surrogate_model,
                  values=np_train_input,
                  output=np_train_output)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  values=np_train_input,
                  output=np_train_output)
    # values not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  output=np_train_output)
    # output not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                  num_points=5,
                  cand_points=np_candidate_points,
                  model=surrogate_model,
                  values=np_train_input)
