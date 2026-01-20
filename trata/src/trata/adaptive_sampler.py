from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from trata import sampler
from scipy import optimize
from copy import deepcopy as cp


class AdaptiveSampler(sampler.ContinuousSampler):

    name = "Adaptive"

    @staticmethod
    def sample_points(**kwargs):
        """Returns a set of new sample points based on some set of states given"""
        pass


class ScoredSampler(AdaptiveSampler):
    """A Base Sampler for adaptive sampling based on score"""

    name = "Scored"

    @classmethod
    def sample_points(cls, **kwargs):
        """
        Creates a set of sample points from a set of candidate points based on an implemented score function.

        Classes that inherit from ScoredSampler must implement _get_score. This function is evaluated at all of
        the candidate points. The points with the top scores are returned. Must specify either nCand and box, or
        npCandPnts. Generated candidate points are created using LatinHyperCubeSampler.
        """

        if 'cand_points' in kwargs and kwargs['cand_points'] is not None:  # Use provided candidate points
            np_candidate_points = kwargs['cand_points']

        # Generate candidate points using LHS
        elif 'num_cand_points' in kwargs and kwargs['num_cand_points'] is not None:
            i_num_candidates = int(kwargs['num_cand_points'])
            ls_box = kwargs.get("box", None)
            seed = kwargs.get("seed", None)
            np_candidate_points = sampler.LatinHyperCubeSampler.sample_points(num_points=i_num_candidates,
                                                                              box=ls_box,
                                                                              seed=seed)
            kwargs['cand_points'] = np_candidate_points

        else:
            raise TypeError("Neither 'cand_points' nor 'num_cand_points' was given")

        i_num_points = int(kwargs['num_points'])

        ls_scores = cls._get_score(**kwargs)  # Inheriting samplers must implement _get_score

        # Pick the points with the top scores (Sort the list and take the last N)
        indices = np.argsort(ls_scores)[-i_num_points:]  # negative index loops around to the end
        return np_candidate_points[indices]

    @staticmethod
    def _get_score(**kwargs):
        raise NotImplementedError


class WeightedSampler(AdaptiveSampler):
    """A Base Sampler for adaptive sampling based on weights"""

    name = "Weighted"

    @classmethod
    def sample_points(cls, **kwargs):
        """
        Creates a set of sample points from a set of candidate points based on an implemented weight function.

        Classes that inherit from WeightedSampler must implement _get_weights. This function is evaluated at all of
        the candidate points. Points are drawn randomly without replacement. A point's probability of being drawn is
        proportional to its weight. Must specify either nCand and box, or npCandPnts. Generated candidate points are
        created using LatinHyperCubeSampler.
        """

        if 'cand_points' in kwargs and kwargs['cand_points'] is not None:
            np_candidate_points = kwargs['cand_points']
        elif 'num_cand_points' in kwargs and kwargs['num_cand_points'] is not None:
            i_num_candidates = int(kwargs['num_cand_points'])
            ls_box = kwargs['box']
            np_candidate_points = sampler.LatinHyperCubeSampler.sample_points(num_points=i_num_candidates,
                                                                              box=ls_box,
                                                                              seed=seed)
            kwargs['cand_points'] = np_candidate_points
        else:
            raise TypeError("Neither 'cand_points' nor 'num_cand_points' was given")

        i_num_points = int(kwargs['num_points'])

        weights = cls._get_weights(**kwargs)  # Inheriting samplers must implement _get_weights()

        weights_sum = weights.sum()

        # Normalize weights to probability, then choose without replacement.
        # The probability of choosing a point is proportional to its weight
        indices = np.random.choice(len(weights), i_num_points, replace=False, p=weights/weights_sum)

        return np_candidate_points[indices]

    @staticmethod
    def _get_weights(**kwargs):
        raise NotImplementedError


class ActiveLearningSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on standard deviation"""

    name = "Active Learning"

    @staticmethod
    def sample_points(num_points, model, num_cand_points=None, box=None, cand_points=None, seed=None, **kwargs):
        """
        Creates a set of sample points based on the standard deviation of the surrogate model.

        Returns the candidate points that has the highest standard deviation in the surrogate model. Requires a
        surrogate model that can give standard deviation point estimate.

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(ActiveLearningSampler, ActiveLearningSampler).sample_points(num_points=num_points,
                                                                                 model=model,
                                                                                 num_cand_points=num_cand_points,
                                                                                 box=box,
                                                                                 cand_points=cand_points,
                                                                                 seed=seed)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the standard deviation at that point

        Args:
            - kwargs:
                * model (Surrogate Model): The trained surrogate model
                * npCandPnts ([[float]]): The set of candidate points

        Returns ([float]):
            - Array of scores
        """

        model = kwargs['model']
        np_candidate_points = kwargs['cand_points']

        # Model must be able to give a standard deviation
        _, np_candidate_sigma = model.predict(np_candidate_points, return_std=True)
        return np_candidate_sigma.flatten()


class BestCandidateSampler(ScoredSampler):
    """Returns the best additional samples to fill in the feature space."""

    name = "Best-Candidate"

    @staticmethod
    def sample_points(num_points, values, num_cand_points=None, box=None, cand_points=None, seed=None, **kwargs):
        """
        Create a set of points to add to existing input data based on distance.

        Args:
            - num_points (int): The number of sample points to return (required)
            - values (numpy array): The existing samples
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - seed: The random seed

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(BestCandidateSampler, BestCandidateSampler).sample_points(num_points=num_points,
                                                                               values=values,
                                                                               num_cand_points=num_cand_points,
                                                                               box=box,
                                                                               cand_points=cand_points,
                                                                               seed=seed)

    def _get_score(**kwargs):
        """
        Score for each point is the delta relative to the nearest neighbor.

        For each point returns the difference between its output and the output of its nearest neighbor.

        Args:
            - kwargs:
                * cand_points ([[float]]): The set of candidate points
                * values (numpy array): The values of the inputs
                * box ([[float]]): The ranges of the inputs

        Returns ([float]):
            - Array of scores
        """
        from sklearn.neighbors import NearestNeighbors

        np_candidate_points = kwargs['cand_points']
        np_values = np.array(kwargs['values'])
        ranges = kwargs['box']

        # Check number of features matches the ranges
        num_features = np_values.shape[1]

        if ranges is not None:
            if len(ranges) != num_features:
                msg = "The number of ranges must match the number of features in existing_data."
                msg += f"You have {len(ranges)} ranges and {num_features} features."
                raise ValueError(msg)

        # Determine the number of nearest neighbors (use 3 or fewer if not enough existing samples).
        n_neighbors = min(3, np_values.shape[0])

        # Use NearestNeighbors to compute distances to the existing dataset.
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(np_values)
        distances, _ = nbrs.kneighbors(np_candidate_points)

        # Compute the average distance to the nearest neighbors for each candidate.
        scores = np.mean(distances, axis=1)
       
        return scores


class DeltaSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on nearest neighbor delta"""

    name = "Delta"

    @staticmethod
    def sample_points(num_points, model, values, output, num_cand_points=None, box=None, cand_points=None, seed=None, **kwargs):
        """
        Creates a set of sample points based on the delta of nearest neighbor outputs.

        Returns the candidate points that has the largest difference between its output and the output of its nearest
        neighbor.

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model
            - values (numpy array): The input points for training 'model'
            - output (numpy array): The output points for training 'model'

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(DeltaSampler, DeltaSampler).sample_points(num_points=num_points,
                                                               model=model,
                                                               num_cand_points=num_cand_points,
                                                               box=box,
                                                               cand_points=cand_points,
                                                               values=values,
                                                               output=output,
                                                               seed=seed)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the delta relative to the nearest neighbor.

        For each point returns the difference between its output and the output of its nearest neighbor.

        Args:
            - kwargs:
                * model (Surrogate Model): The trained surrogate model
                * cand_points ([[float]]): The set of candidate points
                * values (numpy array): The input points for training 'model'
                * output (numpy array): The output points for training 'model'

        Returns ([float]):
            - Array of scores
        """

        model = kwargs['model']
        np_values = kwargs['values']
        np_output = kwargs['output']
        np_candidate_points = kwargs['cand_points']

        np_candidate_predicted = model.predict(np_candidate_points, return_std=False)

        # For each candidate point find the closest point in np_values
        np_closest_indices = np.array([DeltaSampler._get_closest_index(np_candidate_points[i], np_values)
                                       for i in range(np_candidate_points.shape[0])])

        # Get the np_output value from the corresponding np_values value closest to each candidate points
        np_closest_values = np_output[np_closest_indices].flatten()
        return abs(np_closest_values-np_candidate_predicted)#.flatten()

    @staticmethod
    def _get_closest_index(reference_point, values):
        """
        Returns the index of the point closest to the reference point

        Args:
            - reference_point ([float]): The point from which to determine distance
            - values ([[float]]): The set of points on which to determine distance

        Returns (int):
            - The index of the point in np_values closest to the reference point
        """
        np_distances = np.array([np.linalg.norm(reference_point - values[i]) for i in range(values.shape[0])])
        return np_distances.argmin()


class ExpectedImprovementSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on a mix of nearest neighbor delta and standard deviation"""

    name = "Expected Improvement"

    @staticmethod
    def sample_points(num_points, model, values, output, num_cand_points=None, box=None, cand_points=None, seed=None, **kwargs):
        """
        Creates a set of sample points based on the ALM score and Delta score

        Returns the candidate points that maximize the L2 norm of the ALM score and the Delta score

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model
            - values (numpy array): The input points for training 'model'
            - output (numpy array): The output points for training 'model'

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(ExpectedImprovementSampler,
                     ExpectedImprovementSampler).sample_points(num_points=num_points,
                                                               model=model,
                                                               num_cand_points=num_cand_points,
                                                               box=box,
                                                               cand_points=cand_points,
                                                               values=values,
                                                               output=output,
                                                               seed=seed)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the L2 norm of the ALM score and the Delta score.

        .. math::
            \\sqrt{ASM^2 + Delta^2}

        Args:
            - kwargs:
                * model (Surrogate Model): The trained surrogate model
                * npCandPnts ([[float]]): The set of candidate points
                * np_values (numpy array): The input points for training 'model'
                * np_output (numpy array): The output points for training 'model'

        Returns ([float]):
            - Array of scores
        """
        np_delta_score = DeltaSampler._get_score(**kwargs)
        np_active_score = ActiveLearningSampler._get_score(**kwargs)
        return np.sqrt(np.power(np_delta_score, 2.0) + np.power(np_active_score, 2.0))


class LearningExpectedImprovementSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on learning a mix of nearest neighbor delta and standard deviation"""

    name = "Learning Expected Improvement"

    @staticmethod
    def sample_points(num_points, model, values, output, num_cand_points=None, box=None, cand_points=None, seed=None, **kwargs):
        """
        Creates a set of sample points based on learning the best relationship using the ALM score and Delta score

        Learns optimal weights (alpha, beta, rho) for Expected Improvement from training data.
        Scores are based on these learned weights as well as the ALM and Delta scores.

        .. math::
            \\alpha + \\beta(\\rho ALM^2 + (1 - \\rho)Delta^2)

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model
            - values (numpy array): The input points for training 'model'
            - output (numpy array): The output points for training 'model'

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(LearningExpectedImprovementSampler,
                     LearningExpectedImprovementSampler).sample_points(num_points=num_points,
                                                                       model=model,
                                                                       num_cand_points=num_cand_points,
                                                                       box=box,
                                                                       cand_points=cand_points,
                                                                       values=values,
                                                                       output=output,
                                                                       seed=seed)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the optimal combination of ALM and Delta scores learned from the training points.

        Optimizes alpha, beta, and rho on ALM and Delta scores through leave-one-out bootstrapping on the training
        points.

        Args:
            - kwargs:
                * cand_points ([[float]]): The set of candidate points
                * model (Surrogate Model): The trained surrogate model
                * values (numpy array): The input points for training 'model'
                * output (numpy array): The output points for training 'model'

        Returns ([float]):
            - Array of scores
        """

        model = kwargs['model']
        np_values = kwargs['values']
        np_output = kwargs['output']
        np_candidate_points = kwargs['cand_points']

        np_intermediate_ALM, np_intermediate_delta, f_residual = \
            LearningExpectedImprovementSampler._calculate_intermediate_scores(cp(model), np_values, np_output)

        def score(_alpha, _beta, _rho, _alm, _delta):
            return _alpha + _beta * (_rho * np.power(_alm, 2.0) + (1 - _rho) * np.power(_delta, 2.0))

        def error(ls_values):
            _alpha, _beta, _rho = ls_values
            return sum(np.power(f_residual - score(_alpha, _beta, _rho,
                                                   np_intermediate_ALM, np_intermediate_delta),
                                2.0))

        results = [None]*3
        for i, startWeight in enumerate([0.01, 0.5, 0.99]):
            results[i] = optimize.minimize(error,
                                     x0=[f_residual, 1.0, startWeight],
                                     method='L-BFGS-B',
                                     bounds=((None, None), (0.0, None), (0.0, 1.0)))

        ALM_score = ActiveLearningSampler._get_score(model=model, cand_points=np_candidate_points)
        delta_score = DeltaSampler._get_score(model=model, cand_points=np_candidate_points, values=np_values, output=np_output)

        min_index = np.argmin([result['fun'] for result in results])

        alpha, beta, rho = results[min_index].x
        return score(alpha, beta, rho, ALM_score, delta_score)

    @staticmethod
    def _calculate_intermediate_scores(model, values, output):
        """
        Calculates the intermediate scores for each point.

        Fits temporary surrogate models to training points, while leaving each training point out once. The ALM and
        Delta scores are calculated on the left out point. The residual is the difference between the known training
        value and the value predicted when that value is left out.

        Args:
            - model (Surrogate Model): The re-fittable surrogate model
            - values (numpy array): The input points for training 'model'
            - output (numpy array): The output points for training 'model'

        Returns (3-tuple of numpy arrays):
            - Intermediate ALM, Delta, and Residual scores for each point
        """

        np_training_ALM = np.array([0.] * len(output))
        np_training_delta = np.array([0.] * len(output))
        np_residuals = np.array([0.] * len(output))

        for index in range(len(output)):

            # Construct model leaving out np_values[index] and np_output[index]
            left_out_point = values[index].reshape(1, -1)
            np_alt_values = np.delete(values, [index], 0)

            np_alt_output = np.delete(output, [index], 0)

            model.fit(np_alt_values, np_alt_output)

            # Get scores from new model on left out point
            np_training_ALM[index] = ActiveLearningSampler._get_score(model=model, cand_points=left_out_point)[0]
            np_training_delta[index] = DeltaSampler._get_score(model=model, cand_points=left_out_point,
                                                               values=np_alt_values, output=np_alt_output)[0]
            output_pred = model.predict(left_out_point)
            residual_array = np.absolute(output[index] - output_pred)
            np_residuals[index] = residual_array.item()

        np_mean_squared_error = np.power(np_residuals, 2.0)

        np_weight = np.mean(1./np_mean_squared_error)
        f_residual = np.sqrt(1./np_weight)

        return np_training_ALM, np_training_delta, f_residual
