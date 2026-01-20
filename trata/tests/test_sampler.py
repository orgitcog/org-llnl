from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import pytest
import os
import numpy as np
import trata.sampler


def test_LatinHyperCubeSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    ls_test_box_mixed = [[0.0, 25.0], []]
    ls_test_values_mixed = [[], [1, 2, 3]]
    ls_test_values = [['a', 'b', 'c'], [1, 2, 3]]
    np_actual_values_std = trata.sampler.LatinHyperCubeSampler.sample_points(box=ls_test_box,
                                                                                        num_points=25,
                                                                                        seed=2018)
    ls_expected_values_std = [[4.35852039, -5.08143228, -10.43091164],
                              [16.49596893, -20.47257130, -21.89988126],
                              [23.06817236, -6.63575213, -14.80702562],
                              [3.48292099, -11.07181371, 10.72657464],
                              [8.71872672, -24.50730858, 13.15911348],
                              [17.23687915, -7.76359297, 22.14749438],
                              [7.23967253, -3.46858462, 20.81440090],
                              [5.69152361, -16.83474467, -24.13821534],
                              [14.97822306, -23.62861300, -11.40816531],
                              [0.58635238, -4.25774819, -4.02309835],
                              [1.29754221, -12.76295943, 1.37451344],
                              [10.68748533, -1.60803635, 4.07240071],
                              [13.11060267, -14.94061026, 6.01096442],
                              [19.49896516, -15.89723107, -15.42845101],
                              [12.89555267, -18.44274988, 16.13493786],
                              [22.25150919, -13.18069769, 12.24502754],
                              [18.40577694, -8.51018962, 7.50005944],
                              [15.58535840, -2.09769303, 17.10681348],
                              [11.63920998, -22.50995564, 23.19247850],
                              [24.29783886, -9.99696411, -5.97836841],
                              [20.38615112, -19.15633785, -8.05345522],
                              [21.27186961, -17.45631128, -2.96984493],
                              [9.94115246, -0.86130853, 0.31071369],
                              [2.36364119, -10.91427125, -20.96794990],
                              [6.29954539, -21.59838791, -17.91665014]]
    np.testing.assert_array_almost_equal(np_actual_values_std, ls_expected_values_std)
    np_actual_values_geo = trata.sampler.LatinHyperCubeSampler.sample_points(box=ls_test_box,
                                                                                        num_points=25,
                                                                                        geo_degree=1.2,
                                                                                        seed=2018)
    ls_expected_values_geo = [[2.50362419, -3.03469651, -15.08930384],
                              [18.77359544, -22.37383044, -23.46824307],
                              [24.03243512, -4.34533614, -18.90632539],
                              [1.90600791, -9.48422308, 15.38388974],
                              [6.46417686, -24.76583133, 17.57739272],
                              [19.56781870, -5.43265170, 23.59905594],
                              [4.91063990, -1.89665036, 22.88385271],
                              [3.52669501, -19.14878463, -24.59040699],
                              [16.96351106, -24.32851333, -16.02219007],
                              [0.27868428, -2.43051254, -7.64991430],
                              [0.63247481, -14.14349642, 3.77776085],
                              [8.94738347, -0.79650743, 7.71737515],
                              [14.27229556, -16.91204514, 10.23112302],
                              [21.62692165, -18.09780767, -19.37279380],
                              [14.97220416, -20.72503146, 19.88536001],
                              [23.55689222, -14.37890465, 16.77233628],
                              [20.69188967, -6.23322505, 12.00244273],
                              [17.71388978, -1.06094719, 20.58269349],
                              [10.34718887, -23.70865679, 24.14091279],
                              [24.66627359, -8.00295416, -10.19099706],
                              [22.31113137, -21.35061527, -12.61532047],
                              [22.93394050, -19.78645144, -6.20413647],
                              [7.93424954, -0.40936672, 1.94196056],
                              [1.21711691, -9.25769661, -22.97401955],
                              [4.04396729, -23.14706354, -21.11128515]]
    np.testing.assert_array_almost_equal(np_actual_values_geo, ls_expected_values_geo)
    np_actual_values_mixed = trata.sampler.LatinHyperCubeSampler.sample_points(box=ls_test_box_mixed,
                                                                                          values=ls_test_values_mixed,
                                                                                          num_points=5,
                                                                                          seed=2018)
    ls_expected_values_mixed = [[4.53504667, 2],
                                [6.53199449, 3],
                                [17.23204436, 2],
                                [22.94992696, 1],
                                [14.18555550, 1]]
    np.testing.assert_array_almost_equal(np_actual_values_mixed, ls_expected_values_mixed)
    np_actual_values_discrete = trata.sampler.LatinHyperCubeSampler.sample_points(values=ls_test_values,
                                                                                             num_points=5,
                                                                                             seed=2018)
    np_expected_values_discrete = np.array([['a', 2],
                                            ['a', 3],
                                            ['c', 2],
                                            ['c', 1],
                                            ['b', 1]], dtype='O')
    np.testing.assert_array_equal(np_actual_values_discrete, np_expected_values_discrete)

def test_LatinHyperCubeSampler_invalid():
    # nPts not given
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]])

    # neither box nor values given
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      num_points=10)

    # nPts nor box not given
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points)

    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[0.0, 1.0],
                      num_points=10)

    # too many dimensions in nPts
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points=[1, 2])
    # nPts type str
    pytest.raises(ValueError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points='f')
    # box not list
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=1.0,
                      num_points=10)
    # too many dimensions in box
    pytest.raises(TypeError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      num_points=10)
    # geo_degree too low
    pytest.raises(ValueError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points=10,
                      geo_degree=-1.0)

    # geo_degree too high
    pytest.raises(ValueError, trata.sampler.LatinHyperCubeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points=10,
                      geo_degree=3.0)

def test_MonteCarloSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    np_actual_values = trata.sampler.MonteCarloSampler.sample_points(box=ls_test_box,
                                                                                num_points=25,
                                                                                seed=2018)
    ls_expected_values = [[22.05873279, -20.98472622, 18.71205141],
                          [2.60819345, -22.75118388, -0.93975390],
                          [22.67523334, -0.72944103, -18.13007310],
                          [7.65997247, -4.58556066, 9.51107713],
                          [11.16022181, -10.71585676, 0.10592748],
                          [14.74963479, -16.35367116, -21.27444596],
                          [20.92777749, -14.90639990, 1.17561433],
                          [17.44501518, -21.56542405, 20.92838613],
                          [20.07007094, -2.47663773, 1.37143481],
                          [2.68037697, -1.65159673, -6.78760662],
                          [18.92731315, -23.81557158, 21.40931473],
                          [24.99177532, -8.21232790, -0.36542876],
                          [18.14827494, -24.12920345, -13.17964834],
                          [3.53620599, -18.68271599, 1.57076904],
                          [8.91801491, -11.07187365, -16.73723360],
                          [23.56760275, -11.85441308, -6.43065020],
                          [15.25404717, -16.17580535, 12.11259030],
                          [5.68943682, -22.67542577, -13.14797138],
                          [16.71830928, -17.38727558, -5.40181750],
                          [17.32261387, -3.43925346, -22.03051290],
                          [10.42156266, -7.07658654, -19.86155332],
                          [4.29523899, -0.89821269, 2.86250621],
                          [24.42226265, -11.50745340, 15.96511553],
                          [8.25560358, -1.23650456, -0.50948082],
                          [15.72610376, -8.30046097, 20.11534825]]
    np.testing.assert_array_almost_equal(ls_expected_values, np_actual_values)

def test_MonteCarloSampler_invalid():
    # nPts not given
    pytest.raises(TypeError, trata.sampler.MonteCarloSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]])
    # box not given
    pytest.raises(TypeError, trata.sampler.MonteCarloSampler.sample_points,
                      num_points=10)
    # nPts nor box not given
    pytest.raises(TypeError, trata.sampler.MonteCarloSampler.sample_points)

    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.MonteCarloSampler.sample_points,
                      box=[0.0, 1.0],
                      num_points=10)

    # too many dimensions in nPts
    pytest.raises(TypeError, trata.sampler.MonteCarloSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points=[1, 2])

    # nPts type str
    pytest.raises(ValueError, trata.sampler.MonteCarloSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points='f')
    # box not list
    pytest.raises(TypeError, trata.sampler.MonteCarloSampler.sample_points,
                      box=1.0,
                      num_points=10)

    # too many dimensions in box
    pytest.raises(ValueError, trata.sampler.MonteCarloSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      num_points=10)

def test_QuasiRandomNumberSampler_valid():
    """Test basic Sobol and Halton sampling"""
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    
    # Test Sobol
    np_actual_values_sobol = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=3,
        technique='Sobol'
    )
    
    # NOTE: Expected values changed because scipy uses different direction numbers
    # These are the new expected values from scipy's Sobol implementation
    ls_expected_values_sobol = [[0.0, -25.0, -25.0],
                                [12.5, -12.5, 0.0],
                                [18.75, -18.75, -12.5]]
    np.testing.assert_array_almost_equal(np_actual_values_sobol, ls_expected_values_sobol, decimal=5)
    
    # Test Halton
    np_actual_values_halton = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=3,
        technique='Halton'
    )
    
    # NOTE: Expected values changed because scipy's Halton implementation differs
    ls_expected_values_halton = [[0.0, -25.0, -25.0],
                                 [12.5, -16.666666666666668, -15.0],
                                 [6.25, -8.333333333333336, -5.0]]
    np.testing.assert_array_almost_equal(np_actual_values_halton, ls_expected_values_halton, decimal=5)


def test_QuasiRandomNumberSampler_sequence_offset():
    """Test that sequence_offset allows continuing sequences"""
    ls_test_box = [[0.0, 1.0], [0.0, 1.0]]
    
    # Generate 10 points at once
    all_at_once = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=10,
        technique='sobol'
    )
    
    # Generate same 10 points in two batches
    batch1 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=5,
        technique='sobol',
        sequence_offset=0
    )
    
    batch2 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=5,
        technique='sobol',
        sequence_offset=5
    )
    
    # First 5 points should match
    np.testing.assert_array_almost_equal(all_at_once[:5], batch1)
    # Last 5 points should match
    np.testing.assert_array_almost_equal(all_at_once[5:], batch2)


def test_QuasiRandomNumberSampler_scramble_seed():
    """Test that scrambling with seed gives reproducible results"""
    ls_test_box = [[0.0, 1.0], [0.0, 1.0]]
    
    # Generate with scramble and seed
    points1 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=10,
        technique='sobol',
        scramble=True,
        seed=42
    )
    
    # Generate again with same seed
    points2 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=10,
        technique='sobol',
        scramble=True,
        seed=42
    )
    
    # Should be identical
    np.testing.assert_array_equal(points1, points2)
    
    # Generate with different seed
    points3 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=10,
        technique='sobol',
        scramble=True,
        seed=99
    )
    
    # Should be different
    assert not np.array_equal(points1, points3)


def test_QuasiRandomNumberSampler_no_scramble_deterministic():
    """Test that without scrambling, results are deterministic"""
    ls_test_box = [[0.0, 1.0], [0.0, 1.0]]
    
    points1 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=10,
        technique='sobol',
        scramble=False
    )
    
    points2 = trata.sampler.QuasiRandomNumberSampler.sample_points(
        box=ls_test_box,
        num_points=10,
        technique='sobol',
        scramble=False
    )
    
    # Should be identical (no randomness)
    np.testing.assert_array_equal(points1, points2)


def test_QuasiRandomNumberSampler_invalid():
    """Test error handling for invalid inputs"""
    # num_points not given
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=[[0.0, 1.0], [0.0, 1.0]])
    
    # box not given
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  num_points=10)
    
    # neither num_points nor box given
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points)
    
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=[0.0, 1.0],
                  num_points=10)
    
    # too many dimensions in num_points
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=[[0.0, 1.0], [0.0, 1.0]],
                  num_points=[1, 2])
    
    # num_points type str
    pytest.raises(ValueError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=[[0.0, 1.0], [0.0, 1.0]],
                  num_points='f')
    
    # box not list
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=1.0,
                  num_points=10)
    
    # too many dimensions in box
    pytest.raises(TypeError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                  num_points=10)
    
    # invalid sequence type
    pytest.raises(ValueError, trata.sampler.QuasiRandomNumberSampler.sample_points,
                  box=[[0.0, 1.0], [0.0, 1.0]],
                  num_points=10,
                  technique='foobar')

def test_CenteredSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    np_actual_values = trata.sampler.CenteredSampler.sample_points(box=ls_test_box,
                                                                              num_divisions=[3, 5, 6],
                                                                              default=[1, 2, 3])
    np_expected_values = [[0., 2., 3.],
                          [12.5, 2., 3.],
                          [25., 2., 3.],
                          [1., -25., 3.],
                          [1., -18.75, 3.],
                          [1., -12.5, 3.],
                          [1., -6.25, 3.],
                          [1., 0., 3.],
                          [1., 2., -25.],
                          [1., 2., -15.],
                          [1., 2., -5.],
                          [1., 2., 5.],
                          [1., 2., 15.],
                          [1., 2., 25.]]
    np.testing.assert_array_equal(np_actual_values, np_expected_values)

def test_CenteredSampler_invalid():
    # nDiv not given
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      default=[1, 2])
    # box not given
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      num_divisions=10,
                      default=[1, 2])
    # default not given
    pytest.raises(ValueError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_divisions=10)
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      box=[0.0, 1.0],
                      num_divisions=10,
                      default=[1, 2])
    # nDiv type str
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_divisions='ff',
                      default=[1, 2])
    # box not list
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      box=1.0,
                      num_divisions=10,
                      default=[1, 2])
    # too many dimensions in box
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      num_divisions=10,
                      dim_indices=[1],
                      default=[1, 2])
    # default length less than number of dimensions in box
    pytest.raises(ValueError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_divisions=10,
                      default=[2])
    # default length more than number of dimensions in box
    pytest.raises(ValueError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_divisions=10,
                      default=[1, 2, 3])
    # default type str
    pytest.raises(ValueError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_divisions=10,
                      default="ab")
    # default type int
    pytest.raises(TypeError, trata.sampler.CenteredSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_divisions=10,
                      default=2)

def test_OneAtATimeSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    np_actual_values = trata.sampler.OneAtATimeSampler.sample_points(box=ls_test_box,
                                                                     default=[1, 2, 3],
                                                                     do_oat=True,
                                                                     use_low=False,
                                                                     use_high=False,
                                                                     use_default=False)
    ls_expected_values = [[1., 2., 3.],
                          [0., 2., 3.],
                          [25., 2., 3.],
                          [1., -25., 3.],
                          [1., 0., 3.],
                          [1., 2., -25.],
                          [1., 2., 25.]]

    np.testing.assert_array_equal(np_actual_values, ls_expected_values)

def test_OneAtATimeSampler_invalid():
    # box not given
    pytest.raises(TypeError, trata.sampler.OneAtATimeSampler.sample_points,
                      default=[1, 2])
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=[0.0, 1.0],
                      default=[1, 2],
                      do_oat=True)
    # box not list
    pytest.raises(TypeError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=1.0,
                      default=[1, 2],
                      do_oat=True)
    # too many dimensions in box
    pytest.raises(TypeError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      default=[1, 2],
                      do_oat=True)
    # default length less than number of dimensions in box
    pytest.raises(ValueError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      default=[2],
                      use_default=True)
    # default is None when required for do_oat
    pytest.raises(ValueError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      default=None,
                      do_oat=True)
    # default is None when required for use_default
    pytest.raises(ValueError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      default=None,
                      use_default=True)
    # default has too many dimensions
    pytest.raises(ValueError, trata.sampler.OneAtATimeSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      default=[1, 2, 3],  # 3 dims when box has 2
                      use_default=True)

def test_DefaultValueSampler_valid():
    ls_test_default = [-1, 0, 2, np.pi]
    np_actual_values = trata.sampler.DefaultValueSampler.sample_points(num_points=2,
                                                                                  default=ls_test_default)
    ls_expected_values = [[-1., 0., 2., np.pi],
                          [-1., 0., 2., np.pi]]
    np.testing.assert_array_equal(np_actual_values, ls_expected_values)

def test_DefaultValueSampler_invalid():
    # nPts not given
    pytest.raises(TypeError, trata.sampler.DefaultValueSampler.sample_points,
                      default=[1, 2, 3])
    # default not given
    pytest.raises(TypeError, trata.sampler.DefaultValueSampler.sample_points,
                      num_points=5)
    # default and box not same dimension
    pytest.raises(ValueError, trata.sampler.DefaultValueSampler.sample_points,
                      num_points=5,
                      default=[1, 2, 3],
                      box=[[0, 1], [0, 1]])

def test_CornerSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    ls_test_box_mixed = [[0.0, 25.0], []]
    ls_test_values_mixed = [[], [1, 2, 3]]
    ls_test_values = [['a', 'b', 'c'], [1, 2, 3]]
    np_actual_values = trata.sampler.CornerSampler.sample_points(box=ls_test_box,
                                                                            num_points=9)
    ls_expected_values = [[0., -25., -25.],
                          [0., -25., 25.],
                          [0., 0., -25.],
                          [0., 0., 25.],
                          [25., -25., -25.],
                          [25., -25., 25.],
                          [25., 0., -25.],
                          [25., 0., 25.],
                          [0., -25., -25.]]
    np.testing.assert_array_equal(np_actual_values, ls_expected_values)
    np_actual_values_mixed = trata.sampler.CornerSampler.sample_points(box=ls_test_box_mixed,
                                                                                  values=ls_test_values_mixed)
    ls_expected_values_mixed = [[0., 1],
                                [0., 3],
                                [25., 1],
                                [25., 3]]
    np.testing.assert_array_equal(np_actual_values_mixed, ls_expected_values_mixed)
    np_actual_values_discrete = trata.sampler.CornerSampler.sample_points(values=ls_test_values,
                                                                                     num_points=9)
    np_expected_values_discrete = np.array([['a', 1],
                                            ['a', 3],
                                            ['c', 1],
                                            ['c', 3],
                                            ['a', 1],
                                            ['a', 3],
                                            ['c', 1],
                                            ['c', 3],
                                            ['a', 1]], dtype='O')
    np.testing.assert_array_equal(np_actual_values_discrete, np_expected_values_discrete)

def test_CornerSampler_invalid():
    # box not given
    pytest.raises(TypeError, trata.sampler.CornerSampler.sample_points,
                      num_points=10)
    # values nor box not given
    pytest.raises(TypeError, trata.sampler.CornerSampler.sample_points)
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.CornerSampler.sample_points,
                      box=[0.0, 1.0],
                      num_points=10)
    # too many dimensions in nPts
    pytest.raises(TypeError, trata.sampler.CornerSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points=[1, 2])
    # nPts type str
    pytest.raises(ValueError, trata.sampler.CornerSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points='f')
    # box not list
    pytest.raises(TypeError, trata.sampler.CornerSampler.sample_points,
                      box=1.0,
                      num_points=10)

def test_UniformSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    np_actual_values = trata.sampler.UniformSampler.sample_points(box=ls_test_box,
                                                                             num_points=6)
    ls_expected_values = [[0., -25., -25.],
                          [5., -20., -15.],
                          [10., -15., -5.],
                          [15., -10., 5.],
                          [20., -5., 15.],
                          [25., 0., 25.]]
    np.testing.assert_array_equal(np_actual_values, ls_expected_values)
    np_actual_values_equal_area = trata.sampler.UniformSampler.sample_points(box=ls_test_box,
                                                                                        num_points=5,
                                                                                        equal_area_divs=True)
    ls_expected_values_equal_area = [[2.5, -22.5, -20.],
                                     [7.5, -17.5, -10.],
                                     [12.5, -12.5, 0.],
                                     [17.5, -7.5, 10.],
                                     [22.5, -2.5, 20.]]
    np.testing.assert_array_equal(np_actual_values_equal_area, ls_expected_values_equal_area)

def test_UniformSampler_invalid():
    # box not given
    pytest.raises(TypeError, trata.sampler.UniformSampler.sample_points,
                      num_points=10)
    # nDiv not given
    pytest.raises(TypeError, trata.sampler.UniformSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]])
    # nDiv nor box not given
    pytest.raises(TypeError, trata.sampler.UniformSampler.sample_points)
    # not enough dimensions in box
    pytest.raises(ValueError, trata.sampler.UniformSampler.sample_points,
                      box=[0.0, 1.0],
                      num_points=10)
    # box not list
    pytest.raises(ValueError, trata.sampler.UniformSampler.sample_points,
                      box=1.0,
                      num_points=10)
    # too many dimensions in box
    pytest.raises(ValueError, trata.sampler.UniformSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      num_points=10)
    # too many dimensions in nDiv
    pytest.raises(TypeError, trata.sampler.UniformSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]],
                      num_points=[1, 2])

def test_CartesianCrossSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    np_actual_values = trata.sampler.CartesianCrossSampler.sample_points(box=ls_test_box,
                                                                                    num_divisions=3)
    ls_expected_values = [[0.0, -25.0, -25.],
                          [0.0, -25.0, 0.],
                          [0.0, -25.0, 25.],
                          [0.0, -12.5, -25.],
                          [0.0, -12.5, 0.],
                          [0.0, -12.5, 25.],
                          [0.0, 0.0, -25.],
                          [0.0, 0.0, 0.],
                          [0.0, 0.0, 25.],
                          [12.5, -25.0, -25.],
                          [12.5, -25.0, 0.],
                          [12.5, -25.0, 25.],
                          [12.5, -12.5, -25.],
                          [12.5, -12.5, 0.],
                          [12.5, -12.5, 25.],
                          [12.5, 0.0, -25.],
                          [12.5, 0.0, 0.],
                          [12.5, 0.0, 25.],
                          [25.0, -25.0, -25.],
                          [25.0, -25.0, 0.],
                          [25.0, -25.0, 25.],
                          [25.0, -12.5, -25.],
                          [25.0, -12.5, 0.],
                          [25.0, -12.5, 25.],
                          [25.0, 0.0, -25.],
                          [25.0, 0.0, 0.],
                          [25.0, 0.0, 25.]]
    np.testing.assert_array_equal(np_actual_values, ls_expected_values)
    np_actual_values_equal_area = trata.sampler.CartesianCrossSampler.sample_points(box=ls_test_box,
                                                                                               num_divisions=2,
                                                                                               equal_area_divs=True)
    ls_expected_values_equal_area = [[6.25, -18.75, -12.5],
                                     [6.25, -18.75, 12.5],
                                     [6.25, -6.25, -12.5],
                                     [6.25, -6.25, 12.5],
                                     [18.75, -18.75, -12.5],
                                     [18.75, -18.75, 12.5],
                                     [18.75, -6.25, -12.5],
                                     [18.75, -6.25, 12.5]]
    np.testing.assert_array_equal(np_actual_values_equal_area, ls_expected_values_equal_area)
    ls_test_box_mixed = [[0.0, 25.0], []]
    ls_test_values_mixed = [[], [1, 2, 3]]
    np_actual_values_mixed = trata.sampler.CartesianCrossSampler.sample_points(box=ls_test_box_mixed,
                                                                                          values=ls_test_values_mixed,
                                                                                          num_divisions=3)
    ls_expected_values_mixed = [[0.0, 1],
                                [0.0, 2],
                                [0.0, 3],
                                [12.5, 1],
                                [12.5, 2],
                                [12.5, 3],
                                [25.0, 1],
                                [25.0, 2],
                                [25.0, 3]]
    np.testing.assert_array_equal(np_actual_values_mixed, ls_expected_values_mixed)
    ls_test_values = [['a', 'b', 'c'], [1, 2, 3]]
    np_actual_values_discrete = trata.sampler.CartesianCrossSampler.sample_points(num_divisions=3,
                                                                                             values=ls_test_values)
    np_expected_values_discrete = np.array([['a', 1],
                                            ['a', 2],
                                            ['a', 3],
                                            ['b', 1],
                                            ['b', 2],
                                            ['b', 3],
                                            ['c', 1],
                                            ['c', 2],
                                            ['c', 3]], dtype='O')
    np.testing.assert_array_equal(np_actual_values_discrete, np_expected_values_discrete)

def test_CartesianCrossSampler_invalid():
    # box or value not given
    pytest.raises(TypeError, trata.sampler.CartesianCrossSampler.sample_points,
                      num_divisions=10)
    # num_divisions not given
    pytest.raises(TypeError, trata.sampler.CartesianCrossSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]])
    # num_divisions, box, nor values not given
    pytest.raises(TypeError, trata.sampler.CartesianCrossSampler.sample_points)
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.CartesianCrossSampler.sample_points,
                      box=[0.0, 1.0],
                      num_divisions=10)
    # box not list
    pytest.raises(TypeError, trata.sampler.CartesianCrossSampler.sample_points,
                      box=1.0,
                      num_divisions=10)
    # too many dimensions in box
    pytest.raises(ValueError, trata.sampler.CartesianCrossSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      num_divisions=10)

def test_SamplePointsSampler_valid():
    ls_test_input = [(1, 2.3), (2, 3.4)]
    np_actual_values = trata.sampler.SamplePointsSampler.sample_points(samples=ls_test_input)
    np.testing.assert_array_equal(np_actual_values, ls_test_input)

def test_SamplePointsSampler_invalid():
    # samples not given
    pytest.raises(TypeError, trata.sampler.SamplePointsSampler.sample_points)
    # not enough dimensions in  samples
    pytest.raises(TypeError, trata.sampler.SamplePointsSampler.sample_points,
                      samples=[0.0, 1.0])
    # too many dimensions in samples
    pytest.raises(TypeError, trata.sampler.SamplePointsSampler.sample_points,
                      samples=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]])
    # samples not list
    pytest.raises(TypeError, trata.sampler.SamplePointsSampler.sample_points,
                      samples=1)
    # sample with incorrect size
    pytest.raises(TypeError, trata.sampler.SamplePointsSampler.sample_points,
                      samples=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0]])

def test_RejectionSampler_valid():
    ls_test_box = [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    def func(x):
        return np.exp(np.sum(-np.power(x, 2.) / 2.))
    np_actual_values_rejection = trata.sampler.RejectionSampler.sample_points(num_points=25,
                                                                                         box=ls_test_box,
                                                                                         func=func,
                                                                                         seed=2018)
    np_expected_values_rejection = [[0.40057563, -1.25351001, -2.61304318],
                                    [-1.07381617, 0.00961220, 1.27224451],
                                    [-0.61153237, -0.05641841, -0.93891683],
                                    [0.00638401, -1.17238788, -0.18477953],
                                    [-2.24043485, -0.54363168, 1.30650383],
                                    [-0.20162045, -1.57054472, 0.00993160],
                                    [2.73830558, 1.09649763, -0.93251958],
                                    [-0.62149077, -1.95475357, 0.92823968],
                                    [-1.76550807, -0.21535468, -0.73526719],
                                    [-0.52720472, -0.58894374, -1.43066312],
                                    [1.53039716, -0.15851950, -0.64281130],
                                    [-0.28274113, 0.16373812, -1.02166101],
                                    [-0.92093671, 1.38762111, -1.74614927],
                                    [-0.02100135, -0.99098210, 0.20080051],
                                    [-1.44409931, 0.38640898, -1.19540595],
                                    [1.07644660, 1.46602592, -0.02166618],
                                    [0.12550015, 1.07923414, -0.13951586],
                                    [1.19383555, -1.37313410, 0.13410598],
                                    [0.53018472, 1.26069710, -2.47937573],
                                    [-0.08699791, 0.95906146, -0.66407291],
                                    [-0.54729379, 0.62944655, -0.97524317],
                                    [1.45894174, -1.08945671, 0.64928777],
                                    [-0.44666867, 1.45799030, 1.45998521],
                                    [-0.59862535, 0.95229270, -1.75587455],
                                    [-0.27850650, -0.30791447, -1.07258371]]
    np.testing.assert_array_almost_equal(np_actual_values_rejection, np_expected_values_rejection)
    np_actual_values_metropolis = trata.sampler.RejectionSampler.sample_points(num_points=25,
                                                                                          box=ls_test_box,
                                                                                          func=func,
                                                                                          seed=2018,
                                                                                          metropolis=True,
                                                                                          burn_in=1000)
    np_expected_values_metropolis = [[-1.41792483, -2.84079562, -3.58690606],
                                     [-0.59613595, -0.07309829, -0.60564416],
                                     [-1.56889534, -1.26488961, -1.52789929],
                                     [-0.87091969, -2.85938522, -1.19933636],
                                     [-1.50718497, -0.82696786, 0.98850861],
                                     [-2.9158022, -1.22711139, -0.40774782],
                                     [-3.00322143, -0.36339367, 2.61243505],
                                     [-1.12922303, -1.69903612, 1.45579197],
                                     [-1.22556711, 0.29214426, 1.29975049],
                                     [-0.56950507, 2.01672788, 0.96088011],
                                     [0.48219265, -0.15552369, 0.61913659],
                                     [-0.24890541, -1.52077338, 1.31333001],
                                     [0.69354689, -0.99653604, 1.38836756],
                                     [-1.84524373, -0.30884195, 0.81729428],
                                     [3.24987048, -1.22842400, -0.23691424],
                                     [0.23621548, -1.29768201, 1.69258297],
                                     [0.25906199, -1.77343239, 0.81158214],
                                     [0.38043382, -1.78100946, 0.82915394],
                                     [-0.65820766, -2.00797198, 0.93239357],
                                     [0.48609420, -0.75330191, 2.26987343],
                                     [0.85482598, -0.16131444, 1.05467902],
                                     [0.91724278, 1.05577052, 1.47391691],
                                     [1.80611106, 0.08171868, 1.16372164],
                                     [1.01369104, 0.14083339, 2.05100511],
                                     [1.69799361, 1.77466968, 2.34618417]]
    np.testing.assert_array_almost_equal(np_actual_values_metropolis, np_expected_values_metropolis)

def test_RejectionSampler_invalid():
    ls_test_box = [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    def func(x):
        return np.exp(np.sum(-np.power(x, 2.) / 2.))
    # no parameters given
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points)
    # nPts not given
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      box=ls_test_box,
                      func=func)
    # box not given
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      func=func)
    # func not given
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=ls_test_box)
    # too many dimensions in box
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=[[[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]],
                           [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]],
                           [[-5., 5.0], [-5.0, 5.0], [-5.0, 5.0]]],
                      func=func)
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=[-5., 5.],
                      func=func)
    # box not list
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=1,
                      func=func)
    # func not function
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=ls_test_box,
                      func="foo")
    # func does not return scalar
    pytest.raises(ValueError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=ls_test_box,
                      func=lambda x: np.exp(-np.power(x, 2.) / 2.))
    pytest.raises(ValueError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=ls_test_box,
                      func=lambda x: np.exp(-np.power(x, 2.) / 2.),
                      metropolis=True)
    # func returns string
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=ls_test_box,
                      func=lambda x: "bar")
    pytest.raises(TypeError, trata.sampler.RejectionSampler.sample_points,
                      num_points=25,
                      box=ls_test_box,
                      func=lambda x: "bar",
                      metropolis=True)

def test_ProbabilityDensityFunctionSampler_valid():
    ls_test_box = [[0., 1.], [-1., 1.]]
    np_actual_values_normal1 = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
        num_points=5,
        box=ls_test_box,
        dist='norm',
        seed=3)
    np_actual_values_normal2 = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
        num_points=5,
        box=ls_test_box,
        dist='norm',
        seed=3)
    np_expected_values_normal = [[ 0.94715712, -0.17737949],
                                 [ 0.60912746, -0.04137074],
                                 [ 0.52412437, -0.31350034],
                                 [ 0.03412682, -0.02190908],
                                 [ 0.43065295, -0.23860902]]
    np.testing.assert_array_almost_equal(np_actual_values_normal1, np_actual_values_normal2)
    np.testing.assert_array_almost_equal(np_actual_values_normal1, np_expected_values_normal)
    np.testing.assert_array_almost_equal(np_actual_values_normal2, np_expected_values_normal)
    np_actual_values_t1 = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(num_points=5,
                                                                                        num_dim=2,
                                                                                        dist='gamma',
                                                                                        seed=3,
                                                                                        a=[2, 3],
                                                                                        scale=[1, 4])
    np_actual_values_t2 = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(num_points=5,
                                                                                        num_dim=2,
                                                                                        dist='gamma',
                                                                                        seed=3,
                                                                                        a=[2, 3],
                                                                                        scale=[1, 4])
    np_expected_values_t = [[ 5.20633524,  7.84328127],
                            [ 2.29609821, 17.52114328],
                            [ 1.33359579, 26.18369324],
                            [ 1.24934521,  7.48623199],
                            [ 1.61073525,  3.41836996]]
    np.testing.assert_array_almost_equal(np_actual_values_t1, np_actual_values_t2)
    np.testing.assert_array_almost_equal(np_actual_values_t1, np_expected_values_t)
    np.testing.assert_array_almost_equal(np_actual_values_t2, np_expected_values_t)


# Skip for Python 2
# For some reason, scipy's distributions are raising a Value Error instead
#  of a TypeError when 'scale' is the wrong type.
# This is fixed in Python 3
@pytest.mark.skipif(sys.version_info[0] < 3, reason="Not supported for Python 2")
def test_ProbabilityDensityFunctionSampler_invalid():
    """Test that error is raised when neither box nor num_dim provided"""
    with pytest.raises(ValueError, match="Must provide either 'box' or 'num_dim'"):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='uniform'
        )

    """Test that error is raised when both box and num_dim provided"""
    with pytest.raises(ValueError, match="Provide either 'box' or 'num_dim', not both"):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='uniform', box=[[0,1]], num_dim=2
        )

    """Test that error is raised for non-existent distribution"""
    with pytest.raises(ValueError, match="Distribution 'nonexistent' not found in scipy.stats"):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='nonexistent', num_dim=2
        )

    """Test that error is raised when parameter arrays don't match dimensions"""
    with pytest.raises(ValueError, match="Parameter 'loc'.*length"):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='norm', num_dim=3, loc=[0, 1]  # 2 values for 3 dims
        )

    """Test various parameter length mismatches"""
    test_cases = [
        {'scale': [1, 2], 'num_dim': 3},
        {'df': [1, 2, 3], 'num_dim': 2}, 
        {'s': [0.5], 'num_dim': 3},
        {'loc': [0, 1, 2], 'scale': [1, 2], 'num_dim': 2}  # mixed lengths
    ]
    
    for params in test_cases:
        with pytest.raises(ValueError, match="length must match dimensions"):
            trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
                num_points=10, dist='norm', **params
            )

    """Test various invalid box formats"""
    invalid_boxes = [
        [[1]],        # Missing upper bound
        [[1, 2, 3]],  # Too many values per dimension
        "not_a_list", # Wrong type
    ]
    
    for bad_box in invalid_boxes:
        with pytest.raises((ValueError)):
            trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
                num_points=10, dist='uniform', box=bad_box
            )

    """Test that negative num_points is handled"""
    with pytest.raises((ValueError, TypeError)):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=-5, dist='uniform', num_dim=2
        )

    """Test edge case of zero points"""
    result = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
        num_points=0, dist='uniform', num_dim=2
    )
    assert result.shape == (0, 2)

    """Test invalid parameters for specific distributions"""
    # Negative scale for normal distribution
    with pytest.raises((ValueError, RuntimeError)):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='norm', num_dim=2, scale=-1
        )
    
    # Invalid parameters for beta distribution
    with pytest.raises((ValueError, RuntimeError)):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='beta', num_dim=2, a=-1, b=2
        )

    """Test invalid box bounds"""
    # Lower bound greater than upper bound
    with pytest.raises(ValueError, match="scale.*must be positive"):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='uniform', box=[[5, 1]]
        )
        # Check that all values are clipped to the valid range
        assert np.all(result >= 1) and np.all(result <= 5)

        """Test handling of extreme parameter values"""
        # Very large scale
        result = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='norm', num_dim=2, loc=0, scale=1e10
        )
        assert result.shape == (10, 2)
        
        # Very small scale (but positive)
        result = trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='norm', num_dim=2, loc=0, scale=1e-10
        )
        assert result.shape == (10, 2)

    """Test non-numeric parameter values"""
    with pytest.raises(TypeError):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='norm', num_dim=2, loc="not_a_number"
        )

    """Test empty parameter lists"""
    with pytest.raises(ValueError):
        trata.sampler.ProbabilityDensityFunctionSampler.sample_points(
            num_points=10, dist='norm', num_dim=2, loc=[]
        )


def test_MultiNormalSampler_valid():
    np_actual_values = trata.sampler.MultiNormalSampler.sample_points(num_points=25,
                                                                                 mean=[1.0, 2.0, 3.0],
                                                                                 covariance=[[1.0, 0.5, 0.1],
                                                                                             [0.5, 1.0, 0.5],
                                                                                             [0.1, 0.5, 1.0]],
                                                                                 seed=2018)
    np_expected_values = [[2.18507508, 1.33219283, 3.40444004],
                          [2.45580305, 2.79099985, 3.78192789],
                          [0.97809132, 2.42311467, 2.83040529],
                          [0.99830677, 1.67857486, 2.31377346],
                          [2.64062947, 2.69627779, 3.90724736],
                          [-0.63700440, 1.15782608, 2.98721932],
                          [2.07661568, 2.83245411, 3.53719972],
                          [-1.07546486, 1.56953727, 2.35431609],
                          [1.26564231, 1.99785346, 3.18016615],
                          [0.75515235, 3.28813876, 5.66078321],
                          [0.92229847, 2.06972309, 3.69465685],
                          [0.69631836, 1.13603517, 2.63540631],
                          [1.13857746, 1.89814276, 2.64154901],
                          [0.31965689, 1.36832078, 1.76253708],
                          [1.78986522, 2.36816818, 3.78449855],
                          [-0.39596770, 1.76565840, 4.58406723],
                          [1.59020526, 1.93166834, 2.53427698],
                          [1.50373515, 3.88863563, 4.05857294],
                          [1.94032257, 3.02900133, 2.88746379],
                          [0.93033979, 2.53087276, 2.75088314],
                          [1.92030482, 2.04845218, 2.12530156],
                          [-0.35170170, 1.09994845, 3.14959654],
                          [0.73150691, 1.97378189, 3.77789695],
                          [-0.05697331, 1.98179841, 2.83996390],
                          [1.51067421, 2.60407247, 3.03563090]]
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_MultiNormalSampler_invalid():
    # no parameters given
    pytest.raises(TypeError, trata.sampler.MultiNormalSampler.sample_points)
    # nPts not given
    pytest.raises(TypeError, trata.sampler.MultiNormalSampler.sample_points,
                      mean=[1.0, 2.0, 3.0],
                      covariance=[[1.0, 0.5, 0.1],
                                  [0.5, 1.0, 0.5],
                                  [0.1, 0.5, 1.0]])
    # mean and covariance different lengths
    pytest.raises(ValueError, trata.sampler.MultiNormalSampler.sample_points,
                      num_points=25,
                      mean=[1.0, 2.0],
                      covariance=[[1.0, 0.5, 0.1],
                                  [0.5, 1.0, 0.5],
                                  [0.1, 0.5, 1.0]])
    # mean not list
    pytest.raises(ValueError, trata.sampler.MultiNormalSampler.sample_points,
                      num_points=25,
                      mean=1.0,
                      covariance=[[1.0, 0.5, 0.1],
                                  [0.5, 1.0, 0.5],
                                  [0.1, 0.5, 1.0]])
    # mean nested list
    pytest.raises(ValueError, trata.sampler.MultiNormalSampler.sample_points,
                      num_points=25,
                      mean=[[1.0, 0.5, 0.1],
                            [0.5, 1.0, 0.5],
                            [0.1, 0.5, 1.0]],
                      covariance=[[1.0, 0.5, 0.1],
                                  [0.5, 1.0, 0.5],
                                  [0.1, 0.5, 1.0]])
    # covariance double nested list
    pytest.raises(ValueError, trata.sampler.MultiNormalSampler.sample_points,
                      num_points=25,
                      mean=[1.0, 2.0, 3.0],
                      covariance=[[[1.0, 0.5, 0.1],
                                   [0.5, 1.0, 0.5],
                                   [0.1, 0.5, 1.0]],
                                  [[1.0, 0.5, 0.1],
                                   [0.5, 1.0, 0.5],
                                   [0.1, 0.5, 1.0]],
                                  [[1.0, 0.5, 0.1],
                                   [0.5, 1.0, 0.5],
                                   [0.1, 0.5, 1.0]]])
    # covariance not positive semi-definite
    pytest.raises(ValueError, trata.sampler.MultiNormalSampler.sample_points,
                      num_points=25,
                      mean=[1.0, 2.0, 3.0],
                      covariance=[[-1.0, 0.5, 0.1],
                                  [0.5, 1.0, 0.5],
                                  [0.1, 0.5, 1.0]])
    if sys.version_info[0] >= 3:
        # covariance not symmetric
        pytest.warns(RuntimeWarning, trata.sampler.MultiNormalSampler.sample_points,
                         num_points=25,
                         mean=[1.0, 2.0, 3.0],
                         covariance=[[1.0, 0.5, 0.8],
                                     [0.2, 1.0, 0.6],
                                     [0.4, 0.3, 1.0]])

def test_FractionalFactorial_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    ls_test_values2 = [['a','b','c'],[1,2,3],[6,7,8]]

    np.random.seed(3)
    np_actual_values = trata.sampler.FractionalFactorialSampler.sample_points(box=ls_test_box,
                                                                              resolution=3,
                                                                              fraction=None)

    np_expected_values = [[25.0, 0.0, -25.0],
                          [25.0, -25.0, 25.0],
                          [0.0, -25.0, -25.0],
                          [0.0, 0.0, 25.0]]
    np.testing.assert_array_equal(np_actual_values, np_expected_values)

    np.random.seed(3)
    np_actual_values = trata.sampler.FractionalFactorialSampler.sample_points(box=ls_test_box,
                                                                              resolution=None,
                                                                              fraction=1)
    np_expected_values = [[25.0, 0.0, -25.0],
                          [25.0, -25.0, 25.0],
                          [0.0, -25.0, -25.0],
                          [0.0, 0.0, 25.0]]
    np.testing.assert_array_equal(np_actual_values, np_expected_values)

    np.random.seed(3)
    np_actual_values = trata.sampler.FractionalFactorialSampler.sample_points(values=ls_test_values2,
                                                                              resolution=3,
                                                                              fraction=None)
    np_expected_values = np.array([['c', 3, 6],
                                   ['c', 1, 8],
                                   ['a', 1, 6],
                                   ['a', 3, 8]], dtype=object)
    np.testing.assert_array_equal(np_actual_values, np_expected_values)

    np.random.seed(3)
    np_actual_values = trata.sampler.FractionalFactorialSampler.sample_points(values=ls_test_values2,
                                                                              resolution=None,
                                                                              fraction=1)
    np_expected_values = np.array([['c', 3, 6],
                                   ['c', 1, 8],
                                   ['a', 1, 6],
                                   ['a', 3, 8]], dtype=object)
    np.testing.assert_array_equal(np_actual_values, np_expected_values)

def test_FractionalFactorial_invalid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    # neither box nor values given
    pytest.raises(TypeError, trata.sampler.FractionalFactorialSampler.sample_points)
    # box not list
    pytest.raises(TypeError, trata.sampler.FractionalFactorialSampler.sample_points,
                  box=1.5)
    # neither resolution nor fraction given
    pytest.raises(ValueError, trata.sampler.FractionalFactorialSampler.sample_points,
                  box=ls_test_box)
    # wrong resolution
    pytest.raises(ValueError, trata.sampler.FractionalFactorialSampler.sample_points,
                  box=ls_test_box, resolution=5)
    # wrong fraction
    pytest.raises(ValueError, trata.sampler.FractionalFactorialSampler.sample_points,
                  box=ls_test_box, fraction=5)
    # wrong dim, resolution, fraction combination
    pytest.raises(ValueError, trata.sampler.FractionalFactorialSampler.sample_points,
                  box=ls_test_box, resolution=3, fraction=3)

def test_MorrisOneAtATimeSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]

    np_actual_values = trata.sampler.MorrisOneAtATimeSampler.sample_points(box=ls_test_box,
                                                                           seed=3)
    np_expected_values = [[  2.35321606, -14.17182649,  -1.04743509,],
                          [  2.35321606, -14.17182649,   4.1081018 ,],
                          [  2.35321606,  -4.96813837,   4.1081018 ,],
                          [  5.92026266,  -4.96813837,   4.1081018 ]]

    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_MorrisOneAtATimeSampler_invalid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    # box not given
    pytest.raises(TypeError, trata.sampler.MorrisOneAtATimeSampler.sample_points,
                  seed=3)
    # box not list
    pytest.raises(IndexError, trata.sampler.MorrisOneAtATimeSampler.sample_points,
                  box=39.4, seed=3)
    # num paths not int
    pytest.raises(TypeError, trata.sampler.MorrisOneAtATimeSampler.sample_points,
                  box=ls_test_box, num_paths=4.7, seed=3)

def test_SobolIndexSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]

    np_actual_values = trata.sampler.SobolIndexSampler.sample_points(num_points=1,
                                                                     box=ls_test_box,
                                                                     include_second_order=False,
                                                                     seed=3)
    np_expected_values = np.array([[ 17.48514699, -11.6326934 ,  -3.98390363],
                                   [ 10.52434819, -17.34999695,   1.84928672],
                                   [  9.68860264, -14.06343186,  -0.59200223],
                                   [ 21.25512224, -12.39403856,  23.994359  ],
                                   [  9.68860264, -11.6326934 ,  -3.98390363],
                                   [ 21.25512224, -17.34999695,   1.84928672],
                                   [ 17.48514699, -14.06343186,  -3.98390363],
                                   [ 10.52434819, -12.39403856,   1.84928672],
                                   [ 17.48514699, -11.6326934 ,  -0.59200223],
                                   [ 10.52434819, -17.34999695,  23.994359  ]])
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

    np_actual_values = trata.sampler.SobolIndexSampler.sample_points(num_points=1,
                                                                     box=ls_test_box,
                                                                     include_second_order=True,
                                                                     seed=3)
    np_expected_values = np.array([[ 17.48514699, -11.6326934 ,  -3.98390363],
                                   [ 10.52434819, -17.34999695,   1.84928672],
                                   [  9.68860264, -14.06343186,  -0.59200223],
                                   [ 21.25512224, -12.39403856,  23.994359  ],
                                   [  9.68860264, -11.6326934 ,  -3.98390363],
                                   [ 21.25512224, -17.34999695,   1.84928672],
                                   [ 17.48514699, -14.06343186,  -3.98390363],
                                   [ 10.52434819, -12.39403856,   1.84928672],
                                   [ 17.48514699, -11.6326934 ,  -0.59200223],
                                   [ 10.52434819, -17.34999695,  23.994359  ],
                                   [ 17.48514699, -14.06343186,  -0.59200223],
                                   [ 10.52434819, -12.39403856,  23.994359  ],
                                   [  9.68860264, -11.6326934 ,  -0.59200223],
                                   [ 21.25512224, -17.34999695,  23.994359  ],
                                   [  9.68860264, -14.06343186,  -3.98390363],
                                   [ 21.25512224, -12.39403856,   1.84928672]])

    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_SobolIndexSampler_invalid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]

    # required arguments not given
    pytest.raises(TypeError, trata.sampler.SobolIndexSampler.sample_points, seed=3)
    # num_points is wrong type
    pytest.raises(TypeError, trata.sampler.SobolIndexSampler.sample_points, 
                  num_points=2.2, box=ls_test_box)
    # box is not list
    pytest.raises(TypeError, trata.sampler.SobolIndexSampler.sample_points,
                  num_points=1, box=3)

# @pytest.mark.xfail
def test_FaceSampler_valid():
    ls_test_box = [[0.0, 25.0], [-25.0, 0.0], [-25.0, 25.0]]
    np_actual_values = trata.sampler.FaceSampler.sample_points(box=ls_test_box,
                                                                          num_divisions=3)
    ls_expected_values = [[0.0, -25.0, -25.0],
                          [0.0, -25.0, 0.0],
                          [0.0, -25.0, 25.0],
                          [0.0, -12.5, -25.0],
                          [0.0, -12.5, 0.0],
                          [0.0, -12.5, 25.0],
                          [0.0, 0.0, -25.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 25.0],
                          [12.5, -25.0, -25.0],
                          [12.5, -25.0, 0.0],
                          [12.5, -25.0, 25.0],
                          [12.5, -12.5, -25.0],
                          [12.5, -12.5, 0.0],
                          [12.5, -12.5, 25.0],
                          [12.5, 0.0, -25.0],
                          [12.5, 0.0, 0.0],
                          [12.5, 0.0, 25.0],
                          [25.0, -25.0, -25.0],
                          [25.0, -25.0, 0.0],
                          [25.0, -25.0, 25.0],
                          [25.0, -12.5, -25.0],
                          [25.0, -12.5, 0.0],
                          [25.0, -12.5, 25.0],
                          [25.0, 0.0, -25.0],
                          [25.0, 0.0, 0.0],
                          [25.0, 0.0, 25.0]]
    np.testing.assert_array_equal(np_actual_values, ls_expected_values)
    np_actual_values_equal_area = trata.sampler.FaceSampler.sample_points(box=ls_test_box,
                                                                                     num_divisions=2,
                                                                                     equal_area_divs=True)
    ls_expected_values_equal_area = [[0.0, -25.0, -25.0],
                                     [0.0, -25.0, 25.0],
                                     [0.0, 0.0, -25.0],
                                     [0.0, 0.0, 25.0],
                                     [25.0, -25.0, -25.0],
                                     [25.0, -25.0, 25.0],
                                     [25.0, 0.0, -25.0],
                                     [25.0, 0.0, 25.0]]
    np.testing.assert_array_equal(np_actual_values_equal_area, ls_expected_values_equal_area)

def test_FaceSampler_invalid():
    # box not given
    pytest.raises(TypeError, trata.sampler.FaceSampler.sample_points,
                      num_divisions=10)
    # nDiv not given
    pytest.raises(TypeError, trata.sampler.FaceSampler.sample_points,
                      box=[[0.0, 1.0], [0.0, 1.0]])
    # nDiv nor box not given
    pytest.raises(TypeError, trata.sampler.FaceSampler.sample_points)
    # not enough dimensions in box
    pytest.raises(TypeError, trata.sampler.FaceSampler.sample_points,
                      box=[0.0, 1.0],
                      num_divisions=10)
    # box not list
    pytest.raises(TypeError, trata.sampler.FaceSampler.sample_points,
                      box=1.0,
                      num_divisions=10)
    # too many dimensions in box
    pytest.raises(ValueError, trata.sampler.FaceSampler.sample_points,
                      box=[[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]],
                      num_divisions=10)

def test_UserValueSampler_valid():
    points1_tab = os.path.join(os.path.dirname(__file__), "points1.tab")
    np_actual_values1 = trata.sampler.UserValueSampler.sample_points(user_samples_file=points1_tab)
    ls_expected_values1 = [[1.0, 1.0, 1.0],
                           [0.2, 3.0, 4.0],
                           [5.0, 0.6, 7.0],
                           [8.0, 9.0, 0.1],
                           [1.0, 1.0, 1.0]]
    np.testing.assert_array_equal(np_actual_values1, ls_expected_values1)
    assert np_actual_values1.dtype==float
    points2_tab = os.path.join(os.path.dirname(__file__), "points2.tab")
    np_actual_values2 = trata.sampler.UserValueSampler.sample_points(user_samples_file=points2_tab)
    ls_expected_values2 = [['1.0', '1.0', '1.0'],
                           ['.2', '3.0', '4.0'],
                           ['5.0', '.6', '7.0'],
                           ['8.0', '9.0', '.1'],
                           ['1.0', '1.0', '1.0'],
                           ['foo', 'bar', 'zyzzx']]
    np.testing.assert_array_equal(np_actual_values2, ls_expected_values2)

def test_UserValueSampler_invalid():
    # user_samples_file not given
    pytest.raises(TypeError, trata.sampler.UserValueSampler.sample_points)
    # user_samples_file not string
    pytest.raises(RuntimeError, trata.sampler.UserValueSampler.sample_points,
                      user_samples_file=123)
    # user_samples_file points to invalid file
    pytest.raises(RuntimeError, trata.sampler.UserValueSampler.sample_points,
                      user_samples_file="points3.tab")
