import numpy as np
from trata.adaptive_sampler import ActiveLearningSampler, BestCandidateSampler, \
    DeltaSampler, ExpectedImprovementSampler, LearningExpectedImprovementSampler
from trata.composite_samples import *
from kosh.operators.core import KoshOperator


class KoshSampler(KoshOperator):
    """
    Creates a down-sampled dataset if using a discrete sampler, or an additional array
    of samples that will improve the model's predictive ability if using a continuous sampler
    from the adaptive_sampler module.

    """
    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **options):
        """
        :param inputs: Kosh datasets of one or more arrays. The arrays should have features 
        as columns and observations as rows.
        :type inputs: kosh datasets
        :param method: The sampling method; LatinHyperCubeSampler, CornerSampler,
        CartesianCrossSampler, SamplePointsSampler, FractionalFactorialSampler, DeltaSampler,
        ExpectedImprovementSampler, or LearningExpectedImprovementSampler.
        :type method: string
        :param feature_names: A list of the names of the features.
        :type feature_names: [string,]
        :param samples: The list of sample points to use.
        :type samples: [[~]]
        :param values: A set of values for discrete variables.
        :type values: [[~]]
        :param num_points: The number of sample points.
        :type num_points: int
        :param box: The bounding box.
        :type box: [[float]]
        :param seed: Random seed
        :type seed: int
        :param geo_degree:
        :type geo_degree: int
        :param resolution: The specified resolution for the set of points that will be
         a fractional factorial design.
        :type resolution: int
        :param fraction: The specified fraction for the set of points that will be a
         fractional factorial design.
        :type fraction: int
        :param num_cand_points: The number of candidate points to generate.
        :type num_cand_points: int
        :param cand_points: The set of candidate points.
        :type cand_points: [[float]]
        :param model: The trained surrogate model.
        :type model: Surrogate model
        :return: A two dimensional numpy array of sample points.
        :rtype: numpy array
        """

        super(KoshSampler, self).__init__(*args, **options)
        self.options = options

    def operate(self, *inputs, format=None):

        # Read in input kosh datasets into one numpy array
        data = inputs[0][:]
        for input_ in inputs[1:]:
            data = np.append(data, input_[:], axis=0)

        # Convert to Samples object

        # 1. Get feature names
        self.feature_names = self.options.get("feature_names", None)
        data_type = data.dtype
        samples_obj = Samples()

        if self.feature_names:
            # Check feature names are strings and the same length as features in data
            for var, i in zip(features, range(len(self.feature_names))):
                msg = f"Variable name must be a string. Was given {type(var)}"
                assert isinstance(var, str), msg
            msg = f"Length of feature_names is {len(self.feature_names)}. "
            msg += f"Should be {data.shape[1]}."
            assert len(self.feature_names) == data.shape[1], msg
        else:
            name_list = []
            for i in range(data.shape[1]):
                name_list.append(f"Feature{i}")
            self.feature_names = np.array(name_list)

        # 2. Save each column of data as appropriate data type in Samples object
        #    We also save data attributes: range, default
        for i, name in enumerate(self.feature_names):
                low, default, high = np.min(data[:,i]), np.median(data[:,i]), np.max(data[:,i])
                samples_obj.set_continuous_variable(name, low, default, high)
                samples_obj.dt_variables[name].np_points = data[:,i]

        ls_variable_objects = [samples_obj.dt_variables[x] for x in self.feature_names]

        ls_box = []
        ls_default = []
        ls_values = []

        for var, i in zip(ls_variable_objects, range(len(ls_variable_objects))):
            if isinstance(var, ContinuousVariable):
                ls_box.append(var.ls_range)
                ls_default.append(var.f_default)
                ls_values.append([])
            else:
                raise ValueError("Variable was not of type \'ContinuousVariable\'")

        continuous_methods = ['ActiveLearningSampler', 'BestCandidateSampler',
                              'DeltaSampler', 'ExpectedImprovementSampler',
                              'LearningExpectedImprovementSampler']

        method = self.options.get("method")
        num_points = self.options.get("num_points")

        if not np.array([isinstance(x, ContinuousVariable) for x in ls_variable_objects]).any():
            raise TypeError("Must use continuous sampler on continuous variable")

        # Get output data as numpy array
        output = self.options.get("output", None)
        if output:
            output = np.array(output[:])

        model = self.options.get("model", None)
        num_cand_points = self.options.get("num_cand_points", None)
        box = self.options.get("box", [[None]])
        cand_points = self.options.get("cand_points", None)
        seed = self.options.get("seed", None)

        if method == 'ActiveLearningSampler':
            sample_object = ActiveLearningSampler.sample_points(num_points,
                                                                model,
                                                                num_cand_points=num_cand_points,
                                                                box=box,
                                                                cand_points=cand_points,
                                                                seed=seed)

        if method == 'BestCandidateSampler':
            sample_object = BestCandidateSampler.sample_points(num_points,
                                                               values=data,
                                                               num_cand_points=num_cand_points,
                                                               box=box,
                                                               cand_points=cand_points,
                                                               seed=seed)

        if method == 'DeltaSampler':    
            sample_object = DeltaSampler.sample_points(num_points,
                                                       model,
                                                       data,
                                                       output,
                                                       num_cand_points=num_cand_points,
                                                       box=box,
                                                       cand_points=cand_points,
                                                       seed=seed)

        elif method == 'ExpectedImprovementSampler':
            sample_object = ExpectedImprovementSampler.sample_points(num_points,
                                                                     model,
                                                                     data,
                                                                     output,
                                                                     num_cand_points=num_cand_points,
                                                                     box=box,
                                                                     cand_points=cand_points,
                                                                     seed=seed)

        elif method == 'LearningExpectedImprovementSampler':
            sample_object = LearningExpectedImprovementSampler.sample_points(num_points,
                                                                             model,
                                                                             data,
                                                                             output,
                                                                             num_cand_points=num_cand_points,
                                                                             box=box,
                                                                             cand_points=cand_points,
                                                                             seed=seed)


        return sample_object
